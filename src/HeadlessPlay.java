import core.Constants;
import core.Types;
import core.game.Game;
import org.json.JSONArray;
import org.json.JSONObject;
import players.Agent;
import players.ExternalProcessAgent;
import players.external.ExternalBotPayloadBuilder;
import utils.file.IO;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Headless config-driven runner for external self-play and training.
 * Uses the same play.json schema as Play.java but never creates GUI objects.
 */
public class HeadlessPlay {

    public static void main(String[] args) {
        try {
            Constants.VISUALS = false;
            Constants.LOG_STATS = false;

            String configPath = args.length > 0 ? args[0] : "play.json";
            JSONObject config = new IO().readJSON(configPath);
            if (config == null || config.isEmpty()) {
                throw new IllegalStateException("Could not read config for headless run: " + configPath);
            }

            if (config.has("Max Turns Capitals")) {
                Constants.MAX_TURNS_CAPITALS = config.getInt("Max Turns Capitals");
            }
            final int selfplayMaxTurnsCapitals = Constants.MAX_TURNS_CAPITALS;
            if (config.has("Max Actions Per Turn")) {
                Game.MAX_ACTIONS_PER_TURN = Math.max(0, config.getInt("Max Actions Per Turn"));
            }
            if (config.has("Max Actions Per Game")) {
                Game.MAX_ACTIONS_PER_GAME = Math.max(0, config.getInt("Max Actions Per Game"));
            }

            JSONArray playersArray = config.getJSONArray("Players");
            JSONArray tribesArray = config.getJSONArray("Tribes");
            if (playersArray.length() != tribesArray.length()) {
                throw new IllegalStateException("Number of players must equal number of tribes.");
            }

            int nPlayers = playersArray.length();
            Run.PlayerType[] playerTypes = new Run.PlayerType[nPlayers];
            Types.TRIBE[] tribes = new Types.TRIBE[nPlayers];
            for (int i = 0; i < nPlayers; ++i) {
                playerTypes[i] = Run.parsePlayerTypeStr(playersArray.getString(i));
                tribes[i] = Run.parseTribeStr(tribesArray.getString(i));
            }

            ArrayList<ArrayList<String>> externalCommands = Run.parseExternalCommands(config, nPlayers);
            Types.GAME_MODE gameMode = Run.parseGameModeStr(config.getString("Game Mode"));
            Types.MAP_TYPE mapType = config.has("Map Type")
                    ? Run.parseMapTypeStr(config.getString("Map Type"))
                    : Run.DEFAULT_MAP_TYPE;
            Types.MAP_SIZE mapSize = config.has("Map Size")
                    ? Run.parseMapSizeStr(config.getString("Map Size"))
                    : Run.defaultMapSizeForPlayers(nPlayers);
            int gloryTargetScore = Run.parseGloryTargetScore(config);

            long agentSeed = config.getLong("Agents Seed");
            long gameSeed = config.getLong("Game Seed");
            long levelSeed = config.getLong("Level Seed");

            Run.AgentFactoryConfig agentFactoryConfig = Run.AgentFactoryConfig.fromConfig(config);
            ArrayList<Agent> players = buildPlayers(playerTypes, externalCommands, agentSeed, agentFactoryConfig);
            Game game = new Game();

            String runMode = config.getString("Run Mode");
            if (runMode.equalsIgnoreCase("PlayLG")) {
                long resolvedGameSeed = gameSeed == -1 ? System.currentTimeMillis() : gameSeed;
                long resolvedLevelSeed = levelSeed == -1 ? System.currentTimeMillis() + new Random().nextInt() : levelSeed;
                game.init(players, resolvedLevelSeed, tribes, resolvedGameSeed, gameMode, mapSize, mapType, gloryTargetScore);
            } else if (runMode.equalsIgnoreCase("PlayFile")) {
                long resolvedGameSeed = gameSeed == -1 ? System.currentTimeMillis() : gameSeed;
                game.init(players, config.getString("Level File"), resolvedGameSeed, gameMode, gloryTargetScore);
            } else if (runMode.equalsIgnoreCase("Replay")) {
                game.init(players, config.getString("Replay File Name"));
            } else {
                throw new IllegalStateException("Unsupported headless run mode: " + runMode);
            }

            boolean adjudicationEnabled = config.optBoolean("Adjudicate Incomplete Games", false);
            String primaryEndReason = "";
            String matchEndReason = "";
            if (adjudicationEnabled) {
                Game.STOP_WITHOUT_TERMINAL_ON_ACTION_LIMIT = true;
                game.run(null, null, false);
                primaryEndReason = game.getTerminalReason();
                System.out.println("Adjudication: enabled");
                System.out.println("Primary End Reason: " + primaryEndReason);
                if (!game.isGameOver() && game.stoppedWithoutTerminal()) {
                    ArrayList<Agent> nnPlayers = new ArrayList<>(players);
                    ArrayList<Agent> adjudicators = buildAdjudicatorPlayers(nPlayers, config, agentSeed);
                    game.setPlayers(adjudicators);
                    Constants.MAX_TURNS_CAPITALS = Math.max(1, config.optInt("Adjudication Max Turns Capitals", Constants.MAX_TURNS_CAPITALS));
                    Game.MAX_ACTIONS_PER_GAME = Math.max(1, config.optInt("Adjudication Max Actions Per Game", Game.MAX_ACTIONS_PER_GAME));
                    Game.STOP_WITHOUT_TERMINAL_ON_ACTION_LIMIT = false;
                    game.resetStopCounters();
                    long adjudicationStarted = System.currentTimeMillis();
                    game.run(null, null, false);
                    long adjudicationMillis = System.currentTimeMillis() - adjudicationStarted;
                    // Restore self-play cap before interpreting terminal reason: adjudication lowers
                    // Constants.MAX_TURNS_CAPITALS while tick continues from the primary phase, which
                    // otherwise produces misleading strings like "turn 40 of 20".
                    Constants.MAX_TURNS_CAPITALS = selfplayMaxTurnsCapitals;
                    String adjudicationEndReason = game.getTerminalReason();
                    boolean failed = !hasWinner(game);
                    System.out.println("Adjudication Result: " + (failed ? "draw" : "winner"));
                    System.out.println("Adjudication End Reason: " + adjudicationEndReason);
                    System.out.println("Adjudication Sec: " + (adjudicationMillis / 1000.0));
                    JSONObject metadata = new JSONObject();
                    metadata.put("adjudicated", true);
                    metadata.put("primary_end_reason", primaryEndReason);
                    metadata.put("adjudication_end_reason", adjudicationEndReason);
                    metadata.put("adjudication_failed", failed);
                    ExternalBotPayloadBuilder.setExtraResultFields(metadata);
                    for (Agent agent : nnPlayers) {
                        agent.result(game.copyGameState(), 0.0);
                    }
                    ExternalBotPayloadBuilder.setExtraResultFields(null);
                    matchEndReason = adjudicationEndReason;
                } else {
                    game.notifyResults();
                    System.out.println("Adjudication Result: not_needed");
                    System.out.println("Adjudication End Reason: " + primaryEndReason);
                    matchEndReason = primaryEndReason;
                }
            } else {
                Game.STOP_WITHOUT_TERMINAL_ON_ACTION_LIMIT = false;
                game.run(null, null);
                matchEndReason = game.getTerminalReason();
            }
            System.out.println("Match End Reason: " + matchEndReason);
        } catch (Exception e) {
            System.err.println("HeadlessPlay failed: " + e.getMessage());
            System.exit(1);
        }
    }

    private static boolean hasWinner(Game game) {
        Types.RESULT[] results = game.getWinnerStatus();
        for (Types.RESULT result : results) {
            if (result == Types.RESULT.WIN) {
                return true;
            }
        }
        return false;
    }

    private static ArrayList<Agent> buildAdjudicatorPlayers(int nPlayers, JSONObject config, long configuredSeed) {
        String bot = config.optString("Adjudicator Bot", "py/bots/simple_bot.py");
        ArrayList<Agent> players = new ArrayList<>();
        long seed = configuredSeed == -1 ? System.currentTimeMillis() + new Random().nextInt() : configuredSeed;
        for (int i = 0; i < nPlayers; i++) {
            players.add(new ExternalProcessAgent(seed + i, new ArrayList<>(Arrays.asList("python", bot))));
        }
        return players;
    }

    private static ArrayList<Agent> buildPlayers(Run.PlayerType[] playerTypes,
                                                 ArrayList<ArrayList<String>> externalCommands,
                                                 long configuredSeed,
                                                 Run.AgentFactoryConfig agentFactoryConfig) {
        ArrayList<Agent> players = new ArrayList<>();
        long agentSeed = configuredSeed == -1 ? System.currentTimeMillis() + new Random().nextInt() : configuredSeed;
        ArrayList<Integer> allIds = new ArrayList<>();
        for (int i = 0; i < playerTypes.length; ++i) {
            allIds.add(i);
        }

        for (int i = 0; i < playerTypes.length; ++i) {
            Agent agent = Run.getAgent(playerTypes[i], agentSeed, externalCommands.get(i), agentFactoryConfig);
            if (agent == null) {
                throw new IllegalStateException("Failed to construct player " + i);
            }
            agent.setPlayerIDs(i, allIds);
            players.add(agent);
        }
        return players;
    }
}
