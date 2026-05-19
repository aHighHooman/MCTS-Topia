package core.game;

import core.Constants;
import core.Types;
import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.actors.Tribe;
import players.Agent;
import players.HumanAgent;
import utils.*;
import gui.GUI;
import gui.WindowInput;
import utils.stats.AIStats;
import utils.stats.GameplayStats;

import java.util.*;

import static core.Constants.*;
import static core.Types.ACTION.*;

public class Game {
    public static int MAX_ACTIONS_PER_TURN = 80;
    public static int MAX_ACTIONS_PER_GAME = 512;
    public static boolean STOP_WITHOUT_TERMINAL_ON_ACTION_LIMIT = false;
    private static final long GUI_PAUSED_SLEEP_MILLIS = 50L;
    private static final long GUI_IDLE_SLEEP_MILLIS = 5L;

    // State of the game (objects, ticks, etc).
    private GameState gs;

    // GameState objects for players to make decisions
    private GameState[] gameStateObservations;

    // Seed for the game state.
    private long seed;

    //Random number generator for the game.
    private Random rnd;

    // List of players of the game
    private Agent[] players;

    //Number of players of the game.
    private int numPlayers;

    // Is the game paused from the GUI?
    private boolean paused, animationPaused;

    // Indicates whether a loaded save must advance the round counter before processing the next tribe.
    private boolean pendingTickAdvanceOnResume;

    // AI stats for each player.
    private AIStats[] aiStats;

    // Gameplay stats for each player.
    private GameplayStats[] gpStats;
    private int totalActionCounter;
    private String terminalReason;
    private boolean stoppedWithoutTerminal;

    // Short label shown by the GUI for generated/replayed games.
    private String runDescription = "";

    /**
     * Constructor of the game
     */
    public Game() {
    }

    /**
     * Initializes the game. This method does the following:
     * Sets the players of the game, the number of players and their IDs
     * Initializes the array to hold the player game states.
     * Assigns the tribes that will play the game.
     * Creates the level reading it from the file 'filename'.
     * Resets the game so it's ready to start.
     * Turn order: by default, turns run following the order in the tribes array.
     *
     * @param players  Players of the game.
     * @param filename Name of the file with the level information.
     * @param seed     Seed for the game (used only for board generation)
     * @param gameMode Game Mode for this game.
     */
    public void init(ArrayList<Agent> players, String filename, long seed, Types.GAME_MODE gameMode) {
        init(players, filename, seed, gameMode, GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE);
    }

    public void init(ArrayList<Agent> players, String filename, long seed, Types.GAME_MODE gameMode, int gloryTargetScore) {

        //Initiate the bare bones of the main game classes
        this.seed = seed;
        this.rnd = new Random(seed);
        this.gs = new GameState(rnd, gameMode, gloryTargetScore);

        this.gs.init(filename);
        initGameStructures(players, this.gs.getTribes().length);
        pendingTickAdvanceOnResume = false;
        totalActionCounter = 0;
        terminalReason = "";
        stoppedWithoutTerminal = false;
        updateAssignedGameStates(true);
    }

    /**
     * Initializes the game. This method does the following:
     * Sets the players of the game, the number of players and their IDs
     * Initializes the array to hold the player game states.
     * Assigns the tribes that will play the game
     * Generates a new level using the seed levelgen_seed
     * Resets the game so it's ready to start.
     * Turn order: by default, turns run following the order in the tribes array.
     *
     * @param players       Players of the game.
     * @param levelgen_seed Seed for the level generator.
     * @param tribes        Array of tribe types to play with.
     * @param seed          Seed for the game (used only for board generation)
     * @param gameMode      Game Mode for this game.
     */
    public void init(ArrayList<Agent> players, long levelgen_seed, Types.TRIBE[] tribes, long seed, Types.GAME_MODE gameMode) {
        init(players, levelgen_seed, tribes, seed, gameMode,
                GameState.RunDefaults.defaultMapSizeForPlayers(tribes.length),
                GameState.RunDefaults.DEFAULT_MAP_TYPE,
                GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE);
    }

    public void init(ArrayList<Agent> players, long levelgen_seed, Types.TRIBE[] tribes, long seed,
                     Types.GAME_MODE gameMode, Types.MAP_SIZE mapSize, Types.MAP_TYPE mapType, int gloryTargetScore) {

        //Initiate the bare bones of the main game classes
        this.seed = seed;
        this.rnd = new Random(seed);
        this.gs = new GameState(rnd, gameMode, gloryTargetScore);

        this.gs.init(levelgen_seed, tribes, mapSize, mapType);
        initGameStructures(players, tribes);
        pendingTickAdvanceOnResume = false;
        totalActionCounter = 0;
        terminalReason = "";
        stoppedWithoutTerminal = false;
        updateAssignedGameStates(true);
    }

    /**
     * Initializes the game from a savegame file
     *
     * @param players  Players who will play this game.
     * @param fileName savegame
     */
    public void init(ArrayList<Agent> players, String fileName) {

        GameLoader gameLoader = new GameLoader(fileName);
        this.seed = gameLoader.getSeed();
        this.rnd = new Random(seed);
        Tribe[] tribes = gameLoader.getTribes();
        this.gs = new GameState(rnd, gameLoader.getGame_mode(), gameLoader.getGloryTargetScore(),
                tribes, gameLoader.getBoard(), gameLoader.getTick());
        this.gs.setGloryResolutionRound(gameLoader.getGloryResolutionRound());
        this.gs.setGameIsOver(gameLoader.getGameIsOver());
        this.pendingTickAdvanceOnResume = gameLoader.isPendingTickAdvance();
        totalActionCounter = 0;
        terminalReason = "";
        stoppedWithoutTerminal = false;
        initGameStructures(players, tribes.length);
        if (this.gs.isGameOver()) {
            updateAssignedGameStates(true);
        }
    }

    /**
     * Initializes game structures depending on number of players and tribes
     *
     * @param players Players to play this game
     * @param nTribes number of tribes the game is set up to start with. Should be the same as players.size().
     */
    private void initGameStructures(ArrayList<Agent> players, int nTribes) {
        if (players.size() != nTribes) {
            throw new IllegalArgumentException("Number of tribes must equal the number of players. There are "
                    + players.size() + " players for " + nTribes + " tribes in this level.");
        }

        //Create the players and agents to control them
        numPlayers = players.size();
        this.players = new Agent[numPlayers];
        this.aiStats = new AIStats[numPlayers];
        this.gpStats = new GameplayStats[numPlayers];

        ArrayList<Integer> allIds = new ArrayList<>();
        for (int i = 0; i < numPlayers; ++i)
            allIds.add(i);

        for (int i = 0; i < numPlayers; ++i) {
            this.players[i] = players.get(i);
            this.players[i].setPlayerIDs(i, allIds);
            this.aiStats[i] = new AIStats(i);
            this.gpStats[i] = new GameplayStats(i);
        }

        this.gameStateObservations = new GameState[numPlayers];
    }


    /**
     * Initializes game structures depending on number of players and tribes
     *
     * @param players Players to play this game
     * @param tribes  Array of tribe types to play with.
     */
    private void initGameStructures(ArrayList<Agent> players, Types.TRIBE[] tribes) {
        int nTribes = tribes.length;
        if (players.size() != nTribes) {
            throw new IllegalArgumentException("Number of tribes must equal the number of players. There are "
                    + players.size() + " players for " + nTribes + " tribes in this level.");
        }

        //Create the players and agents to control them
        numPlayers = players.size();
        this.players = new Agent[numPlayers];
        this.aiStats = new AIStats[numPlayers];
        this.gpStats = new GameplayStats[numPlayers];

        Tribe[] tribeObjects = gs.getTribes();

        for (int tribeIdx = 0; tribeIdx < tribeObjects.length; ++tribeIdx) {
            ArrayList<Integer> allIds = new ArrayList<>();
            for (int i = 0; i < tribes.length; ++i) {
                allIds.add(i);
            }

            // Player order already defines seat assignment. Re-matching by tribe type breaks
            // duplicate-tribe games by collapsing multiple seats onto the last matching tribe.
            this.players[tribeIdx] = players.get(tribeIdx);
            this.players[tribeIdx].setPlayerIDs(tribeIdx, allIds);
            this.aiStats[tribeIdx] = new AIStats(tribeIdx);
            this.gpStats[tribeIdx] = new GameplayStats(tribeIdx);
        }
        this.gameStateObservations = new GameState[numPlayers];
    }

    /**
     * Runs a game once. Receives frame and window input. If any is null, forces a run with no visuals.
     *
     * @param frame window to draw the game
     * @param wi    input for the window.
     */
    public void run(GUI frame, WindowInput wi) {
        run(frame, wi, true);
    }

    public void run(GUI frame, WindowInput wi, boolean notifyResults) {
        if (frame == null || wi == null)
            VISUALS = false;

        stoppedWithoutTerminal = false;
        if (pendingTickAdvanceOnResume) {
            gs.incTick();
            pendingTickAdvanceOnResume = false;
        }

        boolean firstEnd = true;

        while (frame == null || !frame.isClosed()) {
            if (stoppedWithoutTerminal) {
                break;
            }
            if (!notifyResults && !gs.isGameOver() && reachedConfiguredTurnLimit()) {
                terminalReason = "turn_limit: reached turn " + gs.getTick() + " of " + configuredMaxTurns();
                stoppedWithoutTerminal = true;
                break;
            }
//            System.out.println("Frame closed: " + frame.isClosed());
            // Loop while window is still open, even if the game ended.
            // If not playing with visuals, loop is broken when game's ended.

            boolean gameOver = gameOver();
            // Check end of game
            if (firstEnd && gameOver) {
                if (notifyResults) {
                    terminate();
                }

                firstEnd = false;

                printGameResults();
                if (LOG_STATS) {
                    TreeSet<TribeResult> ranking = getCurrentRanking();
                    for (TribeResult tr : ranking) {
                        int idx = tr.getId();
                        AIStats ais = aiStats[idx];
                        if(VERBOSE) ais.print();
                        GameplayStats gps = gpStats[idx];
                        gps.logGameEnd(tr);
                        if(VERBOSE) {
                            gps.print();
                        }
                    }
                }

                if (!VISUALS || frame == null) {
                    // The game has ended, end the loop if we're running without visuals.
                    break;
                }
            }
            if (!gameOver) {
                tick(frame);
            } else {
                frame.update(getGuiGameState(), null);
                sleepQuietly(GUI_PAUSED_SLEEP_MILLIS);
            }
        }
    }

    /**
     * Ticks the game forward. Asks agents for actions and applies returned actions to obtain the next game state.
     *
     * @param frame GUI of the game
     */
    private void tick(GUI frame) {

//        System.out.println("Tick: " + gs.getTick());
        Tribe[] tribes = gs.getTribes();
        int startPlayerId = gs.getActiveTribeID();
        if (startPlayerId < 0) {
            startPlayerId = 0;
            gs.getBoard().setActiveTribeID(startPlayerId);
        }

        boolean[] processedThisTick = new boolean[numPlayers];
        int currentPlayerId = startPlayerId;
        while (true) {
            if (processedThisTick[currentPlayerId]) {
                break;
            }
            processedThisTick[currentPlayerId] = true;

            Tribe tribe = tribes[currentPlayerId];

            if (tribe.getWinner() != Types.RESULT.INCOMPLETE)
            {
                int nextPlayerId = findNextLivingPlayerId(currentPlayerId);
                gs.getBoard().setActiveTribeID(nextPlayerId);
                if (processedThisTick[nextPlayerId]) {
                    break;
                }
                currentPlayerId = nextPlayerId;
                continue;
            }


            //play the full turn for this player
            processTurn(currentPlayerId, tribe, frame);

            int nextPlayerId = findNextLivingPlayerId(currentPlayerId);
            boolean endsRound = processedThisTick[nextPlayerId];
            gs.getBoard().setActiveTribeID(nextPlayerId);

            // Save Game
            if (Constants.WRITE_SAVEGAMES)
                GameSaver.writeTurnFile(gs, getBoard(), seed, endsRound);

            //it may be that this player won the game, no more playing.
            if (gameOver()) {
                return;
            }

            // Check if game should be paused automatically after this turn
            if (VISUALS && frame != null && frame.pauseAfterTurn()) {
                paused = true;
                frame.setPauseAfterTurn(false);
            }

            if (endsRound) {
                break;
            }

            currentPlayerId = nextPlayerId;
        }

        // Check if game should be paused automatically after this tick
        if (VISUALS && frame != null && frame.pauseAfterTick()) {
            paused = true;
            frame.setPauseAfterTick(false);
        }

        //All turns passed, time to increase the tick.
        gs.incTick();
    }

    private int findNextLivingPlayerId(int currentPlayerId) {
        int nextPlayerId = currentPlayerId;
        do {
            nextPlayerId = (nextPlayerId + 1) % numPlayers;
        } while (nextPlayerId != currentPlayerId &&
                gs.getTribe(nextPlayerId).getWinner() == Types.RESULT.LOSS);
        return nextPlayerId;
    }

    /**
     * Process a turn for a given player. It queries the player for an action until no more
     * actions are available or the player returns a EndTurnAction action.
     *
     * @param playerID ID of the player whose turn is being processed.
     * @param tribe    tribe that corresponds to this player.
     */
    private void processTurn(int playerID, Tribe tribe, GUI frame) {
        //Init the turn for this tribe (stars, unit reset, etc).
        gs.initTurn(tribe);

        //Compute the initial player actions and assign the game states.
        gs.computePlayerActions(tribe);
        updateAssignedGameStates(false);

        //Take the player for this turn
        Agent ag = players[playerID];
        boolean isHumanPlayer = ag instanceof HumanAgent;

        //start the timer to the max duration
        ElapsedCpuTimer ect = new ElapsedCpuTimer();
        ect.setMaxTimeMillis(TURN_TIME_MILLIS);

        // Keep track of time remaining for turn thinking
        long remainingECT = TURN_TIME_MILLIS;

        boolean continueTurn = true;
        int curActionCounter = 0;

        // Timer for action execution, delay introduced from GUI. Another delay is added at the end of the turn to
        // make sure all updates are executed and displayed to humans.
        ElapsedCpuTimer actionDelayTimer = null;
        ElapsedCpuTimer endTurnDelay = null;
        if (VISUALS && frame != null) {
            actionDelayTimer = new ElapsedCpuTimer();
            actionDelayTimer.setMaxTimeMillis(FRAME_DELAY);
        }

        while (frame == null || !frame.isClosed()) {
            // Keep track of action played in this loop, null if no action.
            Action action = null;

            // Check GUI end of turn timer
            if (endTurnDelay != null && endTurnDelay.remainingTimeMillis() <= 0) break;

            if (!paused && !animationPaused) {
                // Action request and execution if turn should be continued
                if (continueTurn) {
                    //noinspection ConstantConditions
                    if ((!VISUALS || frame == null) || actionDelayTimer.remainingTimeMillis() <= 0 || isHumanPlayer) {
                        // Get one action from the player
                        ect.setMaxTimeMillis(remainingECT);  // Reset timer ignoring all other timers or updates
                        action = ag.act(gameStateObservations[playerID], ect);
                        remainingECT = ect.remainingTimeMillis(); // Note down the remaining time to use it for the next iteration

                        if (LOG_STATS && !isHumanPlayer)
                            updateBranchingFactor(aiStats[playerID], gs.getTick(), gameStateObservations[playerID], ag);

                        if(LOG_STATS)
                            updateGameplayStatsMove(gpStats[playerID], action, gameStateObservations[playerID]);

                        curActionCounter++;

                        if (actionDelayTimer != null) {  // Reset action delay timer for next action request
                            actionDelayTimer = new ElapsedCpuTimer();
                            actionDelayTimer.setMaxTimeMillis(FRAME_DELAY);
                        }

                        // Continue this turn if there are still available actions and end turn was not requested.
                        // If the agent is human, let him play for now.
                        continueTurn = !gs.isTurnEnding();
                        if (!isHumanPlayer) {
                            ect.setMaxTimeMillis(remainingECT);
                            boolean timeOut = TURN_TIME_LIMITED && ect.exceededMaxTime();
                            continueTurn &= gs.existAvailableActions(tribe) && !timeOut;
                        }

                        if (!isHumanPlayer && action == null) {
                            System.err.println("Warning: player " + playerID + " returned no action on turn "
                                    + gs.getTick() + "; ending turn.");
                            continueTurn = false;
                        }

                        if (!isHumanPlayer && MAX_ACTIONS_PER_TURN > 0 && curActionCounter >= MAX_ACTIONS_PER_TURN) {
                            System.err.println("Warning: player " + playerID + " exceeded " + MAX_ACTIONS_PER_TURN
                                    + " actions on turn " + gs.getTick() + "; forcing end turn.");
                            continueTurn = false;
                        }

                    }
                } else if (endTurnDelay == null) {
                    // If turn should be ending (and we've not already triggered end turn), the action is automatically EndTurn
                    action = new EndTurn(gs.getActiveTribeID());
                }
            }

            // Update GUI after every iteration
            if (VISUALS && frame != null) {
                frame.update(getGuiGameState(), action);

                // Turn should be ending, start timer for delay of next action and show all updates
                if (action != null && action.getActionType() == END_TURN) {
                    if (isHumanPlayer) break;
                    endTurnDelay = new ElapsedCpuTimer();
                    endTurnDelay.setMaxTimeMillis(FRAME_DELAY);
                }

//                try {
//                    Thread.sleep(10);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
            } else if (action != null && action.getActionType() == END_TURN) { // If no visuals and we should end the turn, just break out of loop here
                break;
            }

            if (action != null && !VISUALS || frame != null && (action != null && !isGuiAnimatedAction(action) ||
                    (action = frame.getAnimatedAction()) != null)) {
                // Play the action in the game and update the available actions list and observations
                // Some actions are animated, the condition above checks if this animation is finished and retrieves
                // the action after all the GUI updates.
                gs.next(action);
                totalActionCounter++;
                gs.computePlayerActions(tribe);
                updateAssignedGameStates(false);
                if (MAX_ACTIONS_PER_GAME > 0 && totalActionCounter >= MAX_ACTIONS_PER_GAME && !gameOver()) {
                    System.err.println("Warning: game exceeded " + MAX_ACTIONS_PER_GAME
                            + " total actions; stopping game.");
                    terminalReason = "action_limit: reached " + MAX_ACTIONS_PER_GAME + " total actions";
                    if (STOP_WITHOUT_TERMINAL_ON_ACTION_LIMIT) {
                        stoppedWithoutTerminal = true;
                    } else {
                        gs.forceDraw();
                    }
                    break;
                }
                if (VISUALS && frame != null && frame.pauseAfterAction()) {
                    paused = true;
                    frame.setPauseAfterAction(false);
                }
            }

            if (gameOver()) {
                break;
            }

            if (VISUALS && frame != null) {
                if (paused) {
                    sleepQuietly(GUI_PAUSED_SLEEP_MILLIS);
                } else if (action == null) {
                    sleepQuietly(GUI_IDLE_SLEEP_MILLIS);
                }
            }
        }

        if(LOG_STATS)
            updateGameplayStatsTurn(gpStats[playerID], gs);

        // Ends the turn for this tribe (units that didn't move heal).
        gs.endTurn(tribe);
    }

    private boolean isGuiAnimatedAction(Action action) {
        Types.ACTION actionType = action.getActionType();
        return actionType == ATTACK || actionType == CONVERT || actionType == HEAL_OTHERS;
    }

    /**
     * Prints the results of the game.
     */
    private void printGameResults() {
        Types.RESULT[] results = getWinnerStatus();
        int[] sc = getScores();
        Tribe[] tribes = gs.getBoard().getTribes();

        TreeSet<TribeResult> ranking = gs.getCurrentRanking();
        System.out.println("Turn " + gs.getTick() + "; Game Results:");
        int rank = 1;
        for (TribeResult tr : ranking) {
            int tribeId = tr.getId();
            Agent ag = players[tribeId];
            String[] agentChunks = ag.getClass().toString().split("\\.");
            String agentName = agentChunks[agentChunks.length - 1];

            System.out.print(" #" + rank + ": Player " + tribeId + " Tribe " + tribes[tribeId].getType()
                    + " (" + agentName + "): " + results[tribeId] + ", " + sc[tribeId] + " points");
            if (VERBOSE) {
                System.out.println("    #tech: " + tr.getNumTechsResearched()
                        + ", #cities: " + tr.getNumCities()
                        + ", production: " + tr.getProduction());
            } else {
                System.out.println();
            }
            rank++;
        }
    }


    /**
     * This method call all agents' end-of-game method for post-processing.
     * Agents receive their final game state and reward
     */
    @SuppressWarnings("UnusedReturnValue")
    private void terminate() {

        Tribe[] tribes = gs.getTribes();
        for (int i = 0; i < numPlayers; i++) {
            Agent ag = players[i];
            ag.result(gs.copy(), tribes[i].getScore());
        }
    }

    private void updateBranchingFactor(AIStats aiStats, int turn, GameState currentGameState, Agent ag) {
        ArrayList<Integer> actionCounts = ag.actionsPerUnit(currentGameState);
        aiStats.addBranchingFactor(turn, actionCounts);
        aiStats.addActionsPerStep(turn, ag.actionsPerGameState(gs));
    }

    /**
     * Updates the gameplay stats after a move
     */
    private void updateGameplayStatsMove(GameplayStats gps, Action played, GameState curGameState)
    {
        if (played == null || curGameState == null) return;
        gps.logAction(played, curGameState.getTick());
    }

    /**
     * Updates the gameplay stats at the end of a trun
     */
    private void updateGameplayStatsTurn(GameplayStats gps, GameState curGameState)
    {
        gps.logGameState(curGameState);
    }

    /**
     * Returns the winning status of all players.
     * @return the winning status of all players.
     */
    public Types.RESULT[] getWinnerStatus()
    {
        //Build the results array
        Tribe[] tribes = gs.getTribes();
        Types.RESULT[] results = new Types.RESULT[numPlayers];
        for (int i = 0; i < numPlayers; i++) {
            Tribe tribe = tribes[i];
            results[i] = tribe.getWinner();
        }
        return results;
    }

    /**
     * Returns the current scores of all players.
     * @return the current scores of all players.
     */
    public int[] getScores()
    {
        //Build the results array
        Tribe[] tribes = gs.getTribes();
        int[] scores = new int[numPlayers];
        for (int i = 0; i < numPlayers; i++) {
            scores[i] = tribes[i].getScore();
        }
        return scores;
    }

    /**
     * Updates the state observations for all players with copies of the
     * current game state, adapted for PO.
     */
    private void updateAssignedGameStates(boolean refreshAllPlayers) {
        if (refreshAllPlayers) {
            for (int i = 0; i < numPlayers; i++) {
                gameStateObservations[i] = getGameState(i);
            }
            return;
        }

        int activePlayer = gs.getActiveTribeID();
        if (activePlayer >= 0 && activePlayer < numPlayers) {
            gameStateObservations[activePlayer] = getGameState(activePlayer);
        }
    }

    /**
     * Returns the game state as seen for the player with the index playerIdx. This game state
     * includes only the observations that are visible if partial observability is enabled.
     * @param playerIdx index of the player for which the game state is generated.
     * @return the game state.
     */
    private GameState getGameState(int playerIdx) {
        return gs.copy(playerIdx);
    }

    private GameState getGuiGameState() {
        if (Constants.GUI_PAN_TO_TRIBE) {
            int activeTribeId = gs.getActiveTribeID();
            if (activeTribeId >= 0 && activeTribeId < numPlayers) {
                return getGameState(activeTribeId);
            }
        }

        if (Constants.GUI_FORCE_FULL_OBS || Constants.PLAY_WITH_FULL_OBS) {
            return getGameState(-1);
        }

        int activeTribeId = gs.getActiveTribeID();
        if (activeTribeId >= 0 && activeTribeId < numPlayers) {
            return getGameState(activeTribeId);
        }
        return getGameState(-1);
    }

    /**
     * Returns the game board.
     * @return the game board.
     */
    public Board getBoard()
    {
        return gs.getBoard();
    }

    /**
     * Returns a full-observation copy of the current game state.
     * @return copied authoritative game state.
     */
    public GameState copyGameState() {
        return gs.copy();
    }

    /**
     * Returns a player-specific copy of the current game state.
     * Use playerIdx = -1 to obtain a full-observation copy.
     * @param playerIdx player index for partial observation.
     * @return copied game state for the requested observer.
     */
    public GameState copyPlayerGameState(int playerIdx) {
        return gs.copy(playerIdx);
    }

    /**
     * Returns the current game tick.
     * @return current tick.
     */
    public int getTick() {
        return gs.getTick();
    }

    public Agent[] getPlayers() {
        return players;
    }

    public void setPlayers(ArrayList<Agent> players) {
        initGameStructures(players, numPlayers);
        updateAssignedGameStates(true);
    }

    public void notifyResults() {
        terminate();
    }

    public boolean stoppedWithoutTerminal() {
        return stoppedWithoutTerminal;
    }

    public int getTotalActionCounter() {
        return totalActionCounter;
    }

    public void resetStopCounters() {
        totalActionCounter = 0;
        stoppedWithoutTerminal = false;
        terminalReason = "";
    }

    /**
     * Method to identify the end of the game. If the game is over, the winner is decided.
     * The winner of a game is determined by TribesConfig.GAME_MODE and TribesConfig.MAX_TURNS
     * @return true if the game has ended, false otherwise.
     */
    boolean gameOver() {
        return gs.gameOver();
    }

    public void setAnimationPaused(boolean p) {
        animationPaused = p;
    }
    public void setPaused(boolean p) {
        paused = p;
    }

    public boolean isPaused() {
        return paused;
    }

    public TreeSet<TribeResult> getCurrentRanking() {
        return gs.getCurrentRanking();
    }

    public String getTerminalReason() {
        if (terminalReason != null && !terminalReason.isEmpty()) {
            return terminalReason;
        }
        if (gs == null || !gs.isGameOver()) {
            return "not_over";
        }

        Types.GAME_MODE mode = gs.getGameMode().getCanonicalMode();
        Types.RESULT[] results = getWinnerStatus();
        int winnerId = -1;
        for (int i = 0; i < results.length; i++) {
            if (results[i] == Types.RESULT.WIN) {
                winnerId = i;
                break;
            }
        }

        if (mode.usesCapitalObjective()) {
            int capitalWinnerId = capitalObjectiveWinnerId();
            if (capitalWinnerId >= 0) {
                return "capital_objective: player " + capitalWinnerId + " controls all capitals";
            }
        }

        int maxTurns = configuredMaxTurns();
        if (maxTurns < Integer.MAX_VALUE && gs.getTick() >= maxTurns) {
            return "turn_limit: reached turn " + gs.getTick() + " of " + maxTurns;
        }
        if (mode.usesEliminationObjective() && winnerId >= 0) {
            return "elimination_objective: player " + winnerId + " is last tribe standing";
        }
        if (mode.usesGloryObjective() && winnerId >= 0) {
            return "glory_objective: player " + winnerId + " won after reaching the glory target";
        }
        if (winnerId >= 0) {
            return "score_ranking: player " + winnerId + " ranked first at game end";
        }
        return "draw";
    }

    private int capitalObjectiveWinnerId() {
        int[] capitals = gs.getBoard().getCapitalIDs();
        for (int i = 0; i < gs.getBoard().getTribes().length; i++) {
            Tribe tribe = gs.getBoard().getTribe(i);
            if (tribe.getWinner() == Types.RESULT.LOSS) {
                continue;
            }
            boolean controlsAllCapitals = true;
            for (int capitalId : capitals) {
                if (!tribe.getCitiesID().contains(capitalId)) {
                    controlsAllCapitals = false;
                    break;
                }
            }
            if (controlsAllCapitals) {
                return i;
            }
        }
        return -1;
    }

    public GameplayStats getGamePlayStats(int id) {
        return gpStats[id];
    }

    public void setRunDescription(String runDescription) {
        this.runDescription = runDescription == null ? "" : runDescription;
    }

    public String getRunDescription() {
        return runDescription;
    }

    public void writeSnapshot() {
        if (gs != null) {
            GameSaver.writeTurnFile(gs, getBoard(), seed, false);
        }
    }

    public void setTurnLimitOverride(int turnLimitOverride) {
        if (gs != null) {
            gs.setTurnLimitOverride(turnLimitOverride);
        }
    }

    private void sleepQuietly(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Returns whether the game has already reached a terminal state.
     * @return true if the game is over.
     */
    public boolean isGameOver() {
        return gs.isGameOver();
    }

    private int configuredMaxTurns() {
        int maxTurns = gs.getGameMode().getMaxTurns();
        if (gs.getTurnLimitOverride() > 0) {
            maxTurns = Math.min(maxTurns, gs.getTurnLimitOverride());
        }
        return maxTurns;
    }

    private boolean reachedConfiguredTurnLimit() {
        int maxTurns = configuredMaxTurns();
        return maxTurns < Integer.MAX_VALUE && gs.getTick() >= maxTurns;
    }
}
