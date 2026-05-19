package core.game;

import core.UnlockRules;
import core.TechnologyTree;
import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.cityactions.Build;
import core.actions.cityactions.factory.CityActionBuilder;
import core.actions.tribeactions.factory.TribeActionBuilder;
import core.actions.unitactions.Move;
import core.actions.unitactions.StepMove;
import core.actions.unitactions.Recover;
import core.actions.unitactions.factory.RecoverFactory;
import core.actions.unitactions.factory.UnitActionBuilder;
import core.actors.*;
import core.actors.units.Unit;
import core.levelgen.LevelGenerator;
import utils.file.IO;
import utils.Vector2d;
import utils.graph.PathNode;
import utils.graph.Pathfinder;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.*;

public class GameState {

    //Game mode
    private Types.GAME_MODE gameMode;
    private Types.MAP_TYPE mapType;
    private int gloryTargetScore;

    // Random generator for the game state.
    private Random rnd;

    // Current tick of the game.
    private int tick = 0;

    // Optional override used by tournament-style runners to stop endless stalemates.
    private int turnLimitOverride = -1;

    // Glory resolves at round end, so once the target is first reached we remember that round.
    private int gloryResolutionRound = -1;

    // Board of the game
    private Board board;

    //Indicates if this tribe can end its turn.
    private boolean[] canEndTurn;

    //Actions per city, unit and tribe. These are computed when computePlayerActions() is called
    private HashMap<Integer, ArrayList<Action>> cityActions;
    private HashMap<Integer, ArrayList<Action>> unitActions;
    private ArrayList<Action> tribeActions;

    //Flags the state to indicate that the turn must end
    private boolean turnMustEnd;

    //Indicates if the game is over.
    private boolean gameIsOver;

    /**
     * This variable indicates if the computed actions in this class are updated.
     * It will take the value of the tribeId for which the actions are computed, and -1 if they are
     * not computed or next() is called (as that makes the computed actions obsolete).
     */
    private int computedActionTribeIdFlag;

    // Indicates if a city is leveling up, which reduces action list to only 2 options
    private boolean levelingUp;

    //Ranking of the game
    private TreeSet<TribeResult> ranking;

    //Constructor.
    public GameState(Random rnd, Types.GAME_MODE gameMode) {
        this(rnd, gameMode, RunDefaults.DEFAULT_GLORY_TARGET_SCORE);
    }

    public GameState(Random rnd, Types.GAME_MODE gameMode, int gloryTargetScore) {
        this.rnd = rnd;
        this.gameMode = gameMode;
        this.mapType = null;
        this.gloryTargetScore = gloryTargetScore;
        computedActionTribeIdFlag = -1;
        this.cityActions = new HashMap<>();
        this.unitActions = new HashMap<>();
        this.tribeActions = new ArrayList<>();
        this.ranking = new TreeSet<>();
        this.turnMustEnd = false;
        this.gameIsOver = false;
    }

    //This Constructor is used when loading from a savegame.
    public GameState(Random rnd, Types.GAME_MODE gameMode, Tribe[] tribes, Board board, int tick){
        this(rnd, gameMode, RunDefaults.DEFAULT_GLORY_TARGET_SCORE);
        this.tick = tick;
        this.board = board;
        board.setTribes(tribes);

        canEndTurn = new boolean[tribes.length];
    }

    public GameState(Random rnd, Types.GAME_MODE gameMode, int gloryTargetScore, Tribe[] tribes, Board board, int tick){
        this(rnd, gameMode, gloryTargetScore);
        this.tick = tick;
        this.board = board;
        board.setTribes(tribes);

        canEndTurn = new boolean[tribes.length];
    }

    /**
     * Initializes the GameState using a level generator.
     */
    void init(long levelgen_seed, Types.TRIBE[] tribes) {
        init(levelgen_seed, tribes, RunDefaults.defaultMapSizeForPlayers(tribes.length), RunDefaults.DEFAULT_MAP_TYPE);
    }

    void init(long levelgen_seed, Types.TRIBE[] tribes, Types.MAP_SIZE mapSize, Types.MAP_TYPE mapType) {
        this.mapType = mapType;
        LevelGenerator levelGen = new LevelGenerator(levelgen_seed);
        levelGen.init(mapSize.getSideLength(), mapType, tribes);
        levelGen.generate();
        String[] lines = levelGen.gelLevelLines();
        initGameState(lines);
    }

    /**
     * Initializes the GameState from a file with the board information.
     */
    void init(String filename) {
        this.mapType = null;
        String[] lines = new IO().readFile(filename);
        initGameState(lines);
    }

    /**
     * Initializes a game state from a series of Strings that determine the initial level disposition
     * @param lines all components for the board in its initial state.
     */
    private void initGameState(String[] lines) {

        LevelLoader ll = new LevelLoader();
        board = ll.buildLevel(lines, rnd);

        Tribe[] tribes = board.getTribes();
        for(Tribe tribe : tribes)
        {
            int startingCityId = tribe.getCitiesID().get(0);
            City c = (City) board.getActor(startingCityId);
            Vector2d cityPos = c.getPosition();
            tribe.clearStartingCapitalView(cityPos.x, cityPos.y, TribesConfig.FIRST_CITY_CLEAR_RANGE, rnd, board);
        }

        board.revealExplorationFromCurrentAssets(rnd);

        canEndTurn = new boolean[tribes.length];

    }

    /**
     * Gets a game actor from its id.
     * @param actorId the id of the actor to retrieve
     * @return the actor, null if the id doesn't correspond to an actor (note that it may have
     * been deleted if the actor was removed from the game).
     */
    public Actor getActor(int actorId)
    {
        return board.getActor(actorId);
    }

    /**
     * Returns the current tick of the game. One tick encompasses a turn for all
     * players in the game.
     * @return current tick of the game.
     */
    public int getTick() {
        return tick;
    }

    /**
     * Increases the tick of the game. One tick encompasses a turn for all players in the game.
     */
    void incTick()
    {
        tick++;
    }

    /**
     * Computes all the actions that a player can take given the current game state.
     * Warning: This method can be expensive. In game loop, its computation sits outside the
     * agent's decision time, but agents can use it on their forward models at real expense.
     * @param tribe Tribe for which actions are being computed.
     */
    void computePlayerActions(Tribe tribe)
    {
        board.setActiveTribeID(tribe.getTribeId());

        if(computedActionTribeIdFlag != -1 && computedActionTribeIdFlag == tribe.getTribeId())
        {
            //Actions already computed and next() hasn't been called. No need to recompute again.
            return;
        }

        computedActionTribeIdFlag = tribe.getTribeId();
        this.cityActions = new HashMap<>();
        this.unitActions = new HashMap<>();
        this.tribeActions = new ArrayList<>();

        if(gameIsOver)
            return; // no actions available if the game is over


        ArrayList<Integer> cities = tribe.getCitiesID();
        ArrayList<Integer> allUnits = new ArrayList<>();
        CityActionBuilder cab = new CityActionBuilder();

        int numCities = cities.size();
        int i = 0;
        levelingUp = false;

        while (!levelingUp && i < numCities)
        {
            int cityId = cities.get(i);
            City c = (City) board.getActor(cityId);
            ArrayList<Action> actions = cab.getActions(this, c);
            levelingUp = cab.cityLevelsUp();

            if(actions.size() > 0)
            {
                if(levelingUp)
                {
                    //We may have already processed other cities. Actions for those should be eliminated.
                    cityActions.clear();
                }
                cityActions.put(cityId, actions);
            }

            if(!levelingUp)
            {
                ArrayList<Integer> unitIds = c.getUnitsID();
                allUnits.addAll(unitIds);
                i++;
            }
        }

        int activeTribeID = board.getActiveTribeID();
        if(levelingUp)
        {
            //A city is levelling up. We're done with this city.
            canEndTurn[activeTribeID] = false;
            return;
        }else{
            canEndTurn[activeTribeID] = true;
        }

        //Add the extra units that don't belong to any city.
        allUnits.addAll(tribe.getExtraUnits());

        //Units!
        UnitActionBuilder uab = new UnitActionBuilder();
        for(Integer unitId : allUnits)
        {
            Unit u = (Unit) board.getActor(unitId);
            ArrayList<Action> actions = uab.getActions(this, u);
            if(actions.size() > 0)
                unitActions.put(unitId, actions);
        }

        //This tribe
        TribeActionBuilder tab = new TribeActionBuilder();
        ArrayList<Action> actions = tab.getActions(this, tribe);
        tribeActions.addAll(actions);
    }

    /**
     * Checks if there are actions that the given tribe can take.
     * @param tribe to check if can execute actions.
     * @return true if actions exist. False if no actions available
     * (that includes if this is not this tribe's turn)
     */
    boolean existAvailableActions(Tribe tribe)
    {
        int tribeId = tribe.getTribeId();
        if(board.getActiveTribeID() != tribeId) //Not sure if this is needed, actually.
            return false;

        //Just one action for a city or a unit makes this question false.
        int nActions = 0;
        for(int cityId : cityActions.keySet())
        {
            nActions += cityActions.get(cityId).size();
            if(nActions>0) return true;
        }
        for(int cityId : unitActions.keySet()) {
            nActions += unitActions.get(cityId).size();
            if(nActions>0) return true;
        }

        //No city or unit actions - if there's only one (EndTurn) tribe action, there are no actions available.
        return tribeActions.size() != 1 || !(tribeActions.get(0).getActionType() == Types.ACTION.END_TURN);
    }

    /**
     * Advances the game state applying a single action received.
     * @param action to be executed in the current game state.
     */
    void next(Action action)
    {
        if(action != null)
        {
            boolean executed = false;
            ActionCommand ac = action.getActionType().getCommand();
            if(ac != null)
                executed = ac.execute(action, this);

            if(!executed && ac != null) {
                System.out.println("Tick: " + this.tick + "; action [" + action + "] couldn't execute?");
                if (action instanceof Move && Boolean.getBoolean("tribes.debug.fail_fast_move")) {
                    String details = describeMoveFailure((Move) action);
                    System.out.println("Tick: " + this.tick + "; move failure details: " + details);
                    throw new IllegalStateException("MOVE failed at tick " + this.tick + " for action [" + action + "]: " + details);
                } else if (action instanceof Build && Boolean.getBoolean("tribes.debug.fail_fast_build")) {
                    String details = describeBuildFailure((Build) action);
                    System.out.println("Tick: " + this.tick + "; build failure details: " + details);
                    throw new IllegalStateException("BUILD failed at tick " + this.tick + " for action [" + action + "]: " + details);
                }
                ac.execute(action, this);
            }

            board.revealExplorationFromCurrentAssets(rnd);
            //new actions may have become available, update the 'dirty' flag
            computedActionTribeIdFlag = -1;
        }
    }

    /**
     * Advances the game state applying a single action received.
     * It may also compute the actions available for the next step.
     * It handles turn change if 'action' is an EndTurn action that can be executed.
     * @param action to be executed in the current game state.
     * @param computeActions true if actions available after action has been executed should be computed.
     */
    public void advance(Action action, boolean computeActions)
    {
        if(action != null)
        {
            boolean executed = false;
            ActionCommand ac = action.getActionType().getCommand();
            if(ac != null)
                executed = ac.execute(action, this);

            if(!executed && ac != null) {
                System.out.println("FM: Action [" + action + "] couldn't execute?");
                if (action instanceof Move && Boolean.getBoolean("tribes.debug.fail_fast_move")) {
                    String details = describeMoveFailure((Move) action);
                    System.out.println("FM: Move failure details: " + details);
                    throw new IllegalStateException("Forward-model MOVE failed for action [" + action + "]: " + details);
                } else if (action instanceof Build && Boolean.getBoolean("tribes.debug.fail_fast_build")) {
                    String details = describeBuildFailure((Build) action);
                    System.out.println("FM: Build failure details: " + details);
                    throw new IllegalStateException("Forward-model BUILD failed for action [" + action + "]: " + details);
                }
                ac.execute(action, this);
                //return false;
            }

            if(executed) {
                //it's an end turn
                if(action.getActionType() == Types.ACTION.END_TURN)
                {
                    //manage the end of this turn.
                    this.endTurn(getActiveTribe());

                    //the game may be over
                    gameOver();

                    //Advance player
                    if(!gameIsOver)
                    {
                        int curActiveTribeId = board.getActiveTribeID();
                        boolean playerFound = false;
                        while(!playerFound)
                        {
                            curActiveTribeId = (curActiveTribeId + 1) % canEndTurn.length;
                            if(board.getTribe(curActiveTribeId).getWinner() != Types.RESULT.LOSS)
                                playerFound = true;

                            if(curActiveTribeId == board.getActiveTribeID()) {
                                throw new IllegalStateException("Forward model could not find a live next player after tribe "
                                        + board.getActiveTribeID() + ". Game should already be over.");
                            }
                        }

                        board.setActiveTribeID(curActiveTribeId);

                        //Start the turn for the next tribe
                        this.initTurn(getActiveTribe());
                    }

                }

                board.revealExplorationFromCurrentAssets(rnd);
                computedActionTribeIdFlag = -1;
                if (computeActions)
                    this.computePlayerActions(getActiveTribe());

            }
            //return true;
        }
        //return false;
    }

    /**
     * Ends this turn. Executes a Recover action on all the units that are not fresh
     * @param tribe tribe whose turn is ending.
     */
    void endTurn(Tribe tribe)
    {
        //For all units that didn't execute any action, a Recover action is executed.
        ArrayList<Integer> allTribeUnits = new ArrayList<>();
        ArrayList<Integer> tribeCities = tribe.getCitiesID();

        //1. Get all units
        for(int cityId : tribeCities)
        {
            City city = (City) getActor(cityId);
            allTribeUnits.addAll(city.getUnitsID());
        }

        //Heal the ones that were in a FRESH state.
        allTribeUnits.addAll(tribe.getExtraUnits());    //Add the extra units that don't belong to a city.
        for(int unitId : allTribeUnits)
        {
            Unit unit = (Unit) getActor(unitId);
            if(unit.getStatus() == Types.TURN_STATUS.FRESH)
            {
                LinkedList<Action> recoverActions = new RecoverFactory().computeActionVariants(unit, this);
                if(recoverActions.size() > 0)
                {
                    Recover recoverAction = (Recover)recoverActions.get(0);
                    ActionCommand ac = recoverAction.getActionType().getCommand();
                    if(ac != null)
                        ac.execute(recoverAction, this);
                }
            }
        }
    }

    /**
     * Inits the turn for this player
     * @param tribe whose turn is starting
     */
    void initTurn(Tribe tribe)
    {
        //Get all cities of this tribe
        ArrayList<Integer> tribeCities = tribe.getCitiesID();
        ArrayList<Integer> allTribeUnits = new ArrayList<>();
        this.setEndTurn(false);

        //1. Compute stars per turn.
        int acumProd = 0;
        for (int cityId : tribeCities) {
            City city = (City) getActor(cityId);

            //Cities with an enemy unit in the city's tile don't generate production.
            boolean produces = !city.isInfiltrated();
            Vector2d cityPos = city.getPosition();
            int unitIDAt = board.getUnitIDAt(cityPos.x, cityPos.y);
            if (unitIDAt > 0) {
                Unit u = (Unit) getActor(unitIDAt);
                produces = (u.getTribeId() == tribe.getTribeId());
            }

            if (produces)
                acumProd += city.getProduction();

            city.setInfiltrated(false);

            allTribeUnits.addAll(city.getUnitsID());

            //All temples grow;
            for(Building b : city.getBuildings())
            {
                if(b.type.isTemple()) {
                    int templePoints = ((Temple) b).newTurn();
                    tribe.addScore(templePoints);
                    city.addPointsWorth(templePoints);
                }
            }
        }

        acumProd += board.getEmbassyIncomeForTribe(tribe.getTribeId());
        acumProd += board.getMarketIncomeForTribe(tribe.getTribeId());

        if(tick == 0)
        {
            tribe.setStars(tribe.getType().getStartingStars());
        }else{
            acumProd = Math.max(0, acumProd); //Never have a negative amount of stars.
            tribe.addStars(acumProd);
        }

        //2. Units: all become available. This needs to be done here as some units may have become
        // pushed during other player's turn.
        allTribeUnits.addAll(tribe.getExtraUnits());    //Add the extra units that don't belong to a city.
        for(int unitId : allTribeUnits)
        {
            Unit unit = (Unit) getActor(unitId);
            unit.markTurnStart();
            if(unit.getStatus() == Types.TURN_STATUS.PUSHED) {
                //Pushed units in the previous turn start as if they moved already.
                unit.setStatus(Types.TURN_STATUS.MOVED);
            } else {
                unit.setStatus(Types.TURN_STATUS.FRESH);
            }
        }

        //3. Update tribe pacifist counter
        tribe.addPacifistCount();
        tribe.setUnitsDisabledNextTurn(false);
    }


    /**
     * Pushes a unit following the game rules. If the unit can't be pushed, destroys it.
     * @param toPush unit to push
     * @param startX initial x position
     * @param startY initial y position.
     */
    public void pushUnit(Unit toPush, int startX, int startY)
    {
        Tribe tribe = getTribe(toPush.getTribeId());
        boolean pushed = board.pushUnit(tribe, toPush, startX, startY, rnd);
        if(!pushed)
        {
            killUnit(toPush);
        }
    }

    /**
     * Kills a unit from the game, removing it from the board, its original city and subtracting game score.
     * @param toKill unit to Kill
     */
    public void killUnit(Unit toKill)
    {
        board.removeUnitFromBoard(toKill);
        City c = (City) getActor(toKill.getCityId());
        Tribe tribe = getTribe(toKill.getTribeId());
        board.removeUnitFromCity(toKill, c, tribe);
        Tribe t = getTribe(toKill.getTribeId());
        t.subtractScore(toKill.getType().getPoints());
    }


    /**
     * Public accessor to the copy() functionality of this state.
     * @return a copy of the current game state.
     */
    public GameState copy() {
        return copy(-1);  // No reduction happening if no index specified
    }

    public GameState copyForPlayer(int playerIdx) {
        return copy(playerIdx);
    }

    /**
     * Creates a deep copy of this game state, given player index. Sets up the game state so that it contains
     * only information available to the given player. If -1, state contains all information.
     * @param playerIdx player index that indicates who is this copy for.
     * @return a copy of this game state.
     */
    GameState copy(int playerIdx)
    {
        GameState copy = new GameState(copyRandom(rnd), this.gameMode, this.gloryTargetScore);
        copy.board = board.copy(playerIdx != -1, playerIdx);
        copy.tick = this.tick;
        copy.mapType = this.mapType;
        copy.turnMustEnd = turnMustEnd;
        copy.gameIsOver = gameIsOver;
        copy.turnLimitOverride = turnLimitOverride;
        copy.gloryResolutionRound = gloryResolutionRound;

        int numTribes = getTribes().length;
        copy.canEndTurn = new boolean[numTribes];
        System.arraycopy(canEndTurn, 0, copy.canEndTurn, 0, numTribes);
        copy.levelingUp = levelingUp;

        copy.computedActionTribeIdFlag = -1;
        copy.cityActions = new HashMap<>();
        copy.unitActions = new HashMap<>();
        copy.tribeActions = new ArrayList<>();

        if (!copy.gameIsOver && copy.getActiveTribe() != null) {
            copy.computePlayerActions(copy.getActiveTribe());
        }

        copy.ranking = new TreeSet<>();
        for (TribeResult tr : ranking) copy.ranking.add(tr.copy());

        return copy;
    }

    private Random copyRandom(Random source) {
        try {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            ObjectOutputStream output = new ObjectOutputStream(buffer);
            output.writeObject(source);
            output.flush();

            ByteArrayInputStream inputBuffer = new ByteArrayInputStream(buffer.toByteArray());
            ObjectInputStream input = new ObjectInputStream(inputBuffer);
            return (Random) input.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new IllegalStateException("Failed to copy game RNG state.", e);
        }
    }

    private String describeMoveFailure(Move move) {
        int unitId = move.getUnitId();
        Vector2d destination = move.getDestination();
        if (destination == null) {
            return "destination=null";
        }

        Unit unit = (Unit) getActor(unitId);
        if (unit == null) {
            return "unit-missing";
        }

        Unit visibleDestinationUnit = board.getObservableUnitAt(unit.getTribeId(), destination.x, destination.y);
        Unit actualDestinationUnit = board.getUnitAt(destination.x, destination.y);
        boolean destinationExplored = getTribe(unit.getTribeId()).isExplored(destination.x, destination.y);
        boolean canMove = unit.canMove();
        boolean hiddenEnemyCloak = board.hasHiddenEnemyCloak(unit.getTribeId(), destination.x, destination.y);
        boolean traversable = board.traversable(destination.x, destination.y, unit.getTribeId());
        Pathfinder pathfinder = new Pathfinder(unit.getPosition(), new StepMove(this, unit));
        ArrayList<PathNode> path = pathfinder.findPathTo(destination);
        GameState observerCopy = copy(unit.getTribeId());
        Unit observerUnit = (Unit) observerCopy.getActor(unitId);
        boolean observerPathExists = false;
        boolean observerHasGeneratedMove = false;
        if (observerUnit != null) {
            Pathfinder observerPathfinder = new Pathfinder(observerUnit.getPosition(), new StepMove(observerCopy, observerUnit));
            observerPathExists = observerPathfinder.findPathTo(destination) != null;
            ArrayList<Action> observerActions = observerCopy.getUnitActions(unitId);
            if (observerActions != null) {
                for (Action observerAction : observerActions) {
                    if (observerAction instanceof Move) {
                        Vector2d observerDestination = ((Move) observerAction).getDestination();
                        if (observerDestination != null && observerDestination.equals(destination)) {
                            observerHasGeneratedMove = true;
                            break;
                        }
                    }
                }
            }
        }

        return "unitType=" + unit.getType() +
                ", status=" + unit.getStatus() +
                ", from=" + unit.getPosition() +
                ", to=" + destination +
                ", destinationExplored=" + destinationExplored +
                ", canMove=" + canMove +
                ", traversable=" + traversable +
                ", pathExists=" + (path != null) +
                ", observerPathExists=" + observerPathExists +
                ", observerHasGeneratedMove=" + observerHasGeneratedMove +
                ", visibleDestinationUnit=" + (visibleDestinationUnit == null ? "null" :
                visibleDestinationUnit.getType() + "#" + visibleDestinationUnit.getActorId() + "@tribe" + visibleDestinationUnit.getTribeId()) +
                ", actualDestinationUnit=" + (actualDestinationUnit == null ? "null" :
                actualDestinationUnit.getType() + "#" + actualDestinationUnit.getActorId() + "@tribe" + actualDestinationUnit.getTribeId()) +
                ", hiddenEnemyCloak=" + hiddenEnemyCloak;
    }

    private String describeBuildFailure(Build build) {
        City city = (City) getActor(build.getCityId());
        if (city == null) {
            return "cityMissing";
        }

        Tribe tribe = getTribe(city.getTribeId());
        Vector2d target = build.getTargetPos();
        Types.BUILDING buildingType = build.getBuildingType();

        return "tribeId=" + city.getTribeId()
                + ", cityId=" + build.getCityId()
                + ", buildingType=" + buildingType
                + ", target=" + target
                + ", stars=" + tribe.getStars()
                + ", cost=" + buildingType.getCost()
                + ", unlocked=" + UnlockRules.isBuildingUnlocked(tribe, buildingType)
                + ", explored=" + tribe.isExplored(target.x, target.y)
                + ", tileCityId=" + board.getCityIdAt(target.x, target.y)
                + ", terrain=" + board.getTerrainAt(target.x, target.y)
                + ", resource=" + board.getResourceAt(target.x, target.y)
                + ", buildingAtTarget=" + board.getBuildingAt(target.x, target.y)
                + ", feasible=" + build.isFeasible(this);
    }


    /**
     * Method to identify the end of the game. If the game is over, the winner is decided.
     * The winner of a game is determined by TribesConfig.GAME_MODE and gameMode.getMaxTurns()
     * @return true if the game has ended, false otherwise.
     */
    boolean gameOver() {
        if (gameIsOver) {
            return true;
        }
        int maxTurns = gameMode.getMaxTurns();
        if (turnLimitOverride > 0) {
            maxTurns = Math.min(maxTurns, turnLimitOverride);
        }
        Types.GAME_MODE canonicalMode = gameMode.getCanonicalMode();
        boolean isEnded = false;
        int[] capitals = board.getCapitalIDs();

        if(canonicalMode.usesCapitalObjective()) {
            //Game over if one tribe controls all capitals
            for (int i = 0; i < canEndTurn.length; ++i) {
                Tribe t = board.getTribe(i);

                //Already lost?
                if (t.getWinner() == Types.RESULT.LOSS)
                    continue;

                boolean winner = true;
                for (int cap : capitals) {
                    if (!t.getCitiesID().contains(cap)) {
                        winner = false;
                        break;
                    }
                }

                if (winner) {
                    //we have a winner: tribe t.
                    isEnded = true;
                    board.getTribe(i).setWinner(Types.RESULT.WIN);
                    break; //no need to go further, all the others have lost the game.
                }

            }
        }

        //Compute the current ranking
        computeGameRanking();

        if(canonicalMode.usesEliminationObjective()) {
            isEnded = countNonLossTribes() <= 1;
        }

        if(canonicalMode.usesGloryObjective()) {
            boolean reachedTarget = false;
            for (TribeResult tr : ranking) {
                if (getTribe(tr.getId()).getScore() >= gloryTargetScore) {
                    reachedTarget = true;
                    break;
                }
            }

            if (reachedTarget) {
                if (gloryResolutionRound == -1) {
                    gloryResolutionRound = tick;
                }
                isEnded = tick > gloryResolutionRound;
            } else {
                gloryResolutionRound = -1;
            }
        }

        //We need to set all the winning conditions for the tribes if the game is over.
        boolean reachedFiniteTurnLimit = maxTurns < Integer.MAX_VALUE && tick >= maxTurns;
        if (reachedFiniteTurnLimit && !isEnded && canonicalMode.usesCapitalObjective())
        {
            forceDraw();
            return true;
        }

        if(isEnded || reachedFiniteTurnLimit)
        {
            boolean first = true;
            for(TribeResult tr : ranking)
            {
                int tribeId = tr.getId();
                Types.RESULT res = first? Types.RESULT.WIN : Types.RESULT.LOSS;
                board.getTribe(tribeId).setWinner (res);
                tr.setResult(res);
                first = false;
            }
            isEnded = true;
        }

        gameIsOver = isEnded;
        return isEnded;
    }

    /**
     * Computes the current game ranking based on the current state of the tribes.
     * Updates the field 'ranking' from GameState
     */
    public void computeGameRanking()
    {
        ranking = new TreeSet<>();
        for(int i = 0; i < canEndTurn.length; ++i)
        {
            Tribe t = board.getTribe(i);
            TribeResult tribeResult = new TribeResult(i, t.getWinner(), t.getScore(), t.getTechTree().getNumResearched(), t.getNumCities(), t.getMaxProduction(this));
            ranking.add(tribeResult);
        }
    }

    /**
     * Returns the current ranking of the game. Ranking are computed at the end of each turn.
     * @return the current ranking of the game.
     */
    public TreeSet<TribeResult> getCurrentRanking() {return ranking;}


    /**
     * Indicates if a given tribe can end its turn. Tribes can't end their turn if a city upgrade is pending.
     * @param tribeId id of the tribe to check
     * @return true if turn can be ended.
     */
    public boolean canEndTurn(int tribeId)
    {
        return canEndTurn[tribeId];
    }

    /**
     * Sets the flag for turning ending to 'endTurn'
     * @param endTurn true if the turn must end
     */
    public void setEndTurn(boolean endTurn)
    {
        turnMustEnd = endTurn;
    }

    /**
     * Indicates if the turn is ending to move to the next player.
     * @return if the turn is ending.
     */
    boolean isTurnEnding()
    {
        return turnMustEnd;
    }

    /**
     * Indicates if at present there's a city leveling up
     * @return if there's a city leveling up
     */
    public boolean isLevelingUp() {
        return levelingUp;
    }

    /**
     * Gets the tribes playing this game.
     * @return the tribes
     */
    public Tribe[] getTribes()
    {
        return board.getTribes();
    }


    /**
     * Returns the game board.
     * @return the game board.
     */
    public Board getBoard()
    {
        return board;
    }


    /**
     * Gets the tribe tribeId playing this game.
     * @param tribeID ID of the tribe to pick
     * @return the tribe with the ID requested
     */
    public Tribe getTribe(int tribeID)
    {
        return board.getTribes()[tribeID];
    }

    /**
     * Returns the tribe which turn it is now (the active tribe)
     * @return Current tribe to move.
     */
    public Tribe getActiveTribe() {
        int activeTribeID = board.getActiveTribeID();
        if (activeTribeID != -1) {
            return board.getTribe(activeTribeID);
        } else return null;
    }

    public int getActiveTribeID() {
        return board.getActiveTribeID();
    }

    public Random getRandomGenerator() {
        return rnd;
    }

    public void setTurnLimitOverride(int turnLimitOverride) {
        this.turnLimitOverride = turnLimitOverride > 0 ? turnLimitOverride : -1;
    }

    public int getTurnLimitOverride() {
        return turnLimitOverride;
    }

    boolean isNative() {
        return board.isNative();
    }

    /* AVAILABLE ACTIONS */

    public boolean isGameOver() {
        return gameIsOver;
    }

    void setGameIsOver(boolean gameIsOver) {
        this.gameIsOver = gameIsOver;
    }

    void forceDraw() {
        computeGameRanking();
        for (Tribe tribe : board.getTribes()) {
            if (tribe.getWinner() != Types.RESULT.LOSS) {
                tribe.setWinner(Types.RESULT.INCOMPLETE);
            }
        }
        gameIsOver = true;
    }

    /**
     * Gathers and returns all the available actions for the active tribe in a single ArrayList
     * @return all available actions
     */
    public ArrayList<Action> getAllAvailableActions()
    {
        ArrayList<Action> allActions = new ArrayList<>(this.getTribeActions());
        for (Integer cityId : this.getCityActions().keySet())
        {
            allActions.addAll(this.getCityActions(cityId));
        }
        for (Integer unitId : this.getUnitActions().keySet())
        {
            allActions.addAll(this.getUnitActions(unitId));
        }
        return allActions;
    }

    public ArrayList<Action> getAllAvailableActions(int playerID)
    {
        if(playerID == getActiveTribeID()) {
            return getAllAvailableActions();
        }

        GameState copy = copy(-1);
        copy.getBoard().setActiveTribeID(playerID);
        copy.computedActionTribeIdFlag = -1;
        copy.computePlayerActions(copy.getTribe(playerID));
        return copy.getAllAvailableActions();
    }

    public HashMap<Integer, ArrayList<Action>> getCityActions() {     return cityActions;  }
    public ArrayList<Action> getCityActions(City c) {  return cityActions.get(c.getActorId());  }
    public ArrayList<Action> getCityActions(int cityId) {  return cityActions.get(cityId);  }
    public ArrayList<Action> getAllCityActions()
    {
        ArrayList<Action> allActions = new ArrayList<>();
        for (Integer cityId : this.getCityActions().keySet())
            allActions.addAll(this.getCityActions(cityId));

        return allActions;
    }

    public HashMap<Integer, ArrayList<Action>> getUnitActions() {  return unitActions;  }
    public ArrayList<Action> getUnitActions(int unitId) {  return unitActions.get(unitId);  }
    public ArrayList<Action> getUnitActions(Unit u) {  return unitActions.get(u.getActorId());  }
    public ArrayList<Action> getAllUnitActions()
    {
        ArrayList<Action> allActions = new ArrayList<>();
        for (Integer unitId : this.getUnitActions().keySet())
            allActions.addAll(this.getUnitActions(unitId));

        return allActions;
    }

    public ArrayList<Action> getTribeActions() {  return tribeActions;  }

    /* Potentially helpful methods for agents */

    public int getTribeProduction(int playerId)
    {
        if(playerId == getActiveTribeID())
            return this.getActiveTribe().getMaxProduction(this);

        return this.getTribe(playerId).getMaxProduction(this);
    }


    public TechnologyTree getTribeTechTree(int playerId)
    {
        if(playerId == getActiveTribeID())
            return getActiveTribe().getTechTree();
        return this.getTribe(playerId).getTechTree();
    }

    public int getNKills(int playerId)
    {
        return getTribe(playerId).getnKills();
    }

    public int getScore(int playerID)
    {
        return getTribe(playerID).getScore();
    }

    public boolean[][] getVisibilityMap() {
        return getActiveTribe().getObsGrid();
    }

    public ArrayList<Integer> getTribesMet() {
        return getActiveTribe().getTribesMet();
    }

    public HashMap<Integer, Vector2d> getKnownCapitalPositions() {
        return getKnownCapitalPositions(getActiveTribeID());
    }

    public HashMap<Integer, Vector2d> getKnownCapitalPositions(int playerId) {
        HashMap<Integer, Vector2d> knownCapitals = new HashMap<>();
        Tribe observer = getTribe(playerId);

        for (int tribeId = 0; tribeId < getTribes().length; tribeId++) {
            Vector2d capitalPos = board.getCapitalPosition(tribeId);
            if (capitalPos == null) {
                continue;
            }

            if (tribeId == playerId || observer.getKnownCapitalTribes().contains(tribeId)) {
                knownCapitals.put(tribeId, capitalPos);
            }
        }

        return knownCapitals;
    }

    public Types.RELATIONSHIP[][] getVisibleRelationships(int playerId) {
        Types.RELATIONSHIP[][] visibleRelationships =
                new Types.RELATIONSHIP[getTribes().length][getTribes().length];
        Tribe observer = getTribe(playerId);

        if (observer.getTechTree().isResearched(Types.TECHNOLOGY.DIPLOMACY)) {
            return board.getDiplomacy().copyRelationshipStatus();
        }

        for (int tribeId = 0; tribeId < getTribes().length; tribeId++) {
            visibleRelationships[playerId][tribeId] = board.getDiplomacy().getRelationship(playerId, tribeId);
            visibleRelationships[tribeId][playerId] = board.getDiplomacy().getRelationship(tribeId, playerId);
        }
        return visibleRelationships;
    }

    public ArrayList<City> getCities(int playerId)
    {
        ArrayList<Integer> cities = getTribe(playerId).getCitiesID();
        ArrayList<City> cityActors = new ArrayList<>();
        for(Integer cityId : cities)
        {
            cityActors.add((City)board.getActor(cityId));
        }
        return cityActors;
    }

    public ArrayList<Unit> getUnits(int playerId)
    {
        ArrayList<Integer> cities = getTribe(playerId).getCitiesID();
        ArrayList<Unit> unitActors = new ArrayList<>();
        for(Integer cityId : cities)
        {
            City c = (City)board.getActor(cityId);
            for(Integer unitId : c.getUnitsID())
            {
                Unit unit = (Unit) board.getActor(unitId);
                unitActors.add(unit);
            }
        }

        for(Integer unitId : getTribe(playerId).getExtraUnits())
        {
            Unit unit = (Unit) board.getActor(unitId);
            unitActors.add(unit);
        }

        return unitActors;
    }

    public Types.GAME_MODE getGameMode() {
        return gameMode;
    }

    public Types.MAP_TYPE getMapType() {
        return mapType;
    }

    public void setMapType(Types.MAP_TYPE mapType) {
        this.mapType = mapType;
    }

    public int getGloryTargetScore() {
        return gloryTargetScore;
    }

    int getGloryResolutionRound() {
        return gloryResolutionRound;
    }

    void setGloryResolutionRound(int gloryResolutionRound) {
        this.gloryResolutionRound = gloryResolutionRound;
    }

    private int countNonLossTribes() {
        int numNonLoss = 0;
        for (TribeResult tr : ranking) {
            if (getTribe(tr.getId()).getWinner() != Types.RESULT.LOSS) {
                numNonLoss++;
            }
        }
        return numNonLoss;
    }

    static final class RunDefaults {
        static final int DEFAULT_GLORY_TARGET_SCORE = 10000;

        static Types.MAP_SIZE defaultMapSizeForPlayers(int nPlayers) {
            if (nPlayers <= 2) return Types.MAP_SIZE.TINY;
            if (nPlayers == 3) return Types.MAP_SIZE.SMALL;
            if (nPlayers == 4) return Types.MAP_SIZE.NORMAL;
            if (nPlayers == 5) return Types.MAP_SIZE.LARGE;
            if (nPlayers == 6) return Types.MAP_SIZE.HUGE;
            return Types.MAP_SIZE.MASSIVE;
        }

        static final Types.MAP_TYPE DEFAULT_MAP_TYPE = Types.MAP_TYPE.CONTINENTS;
    }

    public Types.RESULT getTribeWinStatus(int playerId) {
        return getTribe(playerId).getWinner();
    }


}
