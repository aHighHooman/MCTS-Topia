import core.Constants;
import core.Types;
import core.game.Game;
import org.json.JSONArray;
import org.json.JSONObject;
import players.*;
import utils.file.IO;

import java.util.*;

/**
 * Entry point of the framework.
 */
public class Play {

    public static void main(String[] args) {

        try {
            JSONObject config = new IO().readJSON("play.json");

            if (config != null && !config.isEmpty()) {
                String runMode = config.getString("Run Mode");
                Constants.VERBOSE = config.optBoolean("Verbose", Constants.VERBOSE);
                Constants.FRAME_DELAY = Math.max(0, config.optInt("Frame Delay", Constants.FRAME_DELAY));
                Constants.GUI_PAN_TO_TRIBE = config.optBoolean("Follow Active Tribe", Constants.GUI_PAN_TO_TRIBE);
                if (config.has("Max Turns Capitals")) {
                    Constants.MAX_TURNS_CAPITALS = Math.max(1, config.getInt("Max Turns Capitals"));
                }
                if (config.has("Max Actions Per Turn")) {
                    Game.MAX_ACTIONS_PER_TURN = Math.max(0, config.getInt("Max Actions Per Turn"));
                }
                if (config.has("Max Actions Per Game")) {
                    Game.MAX_ACTIONS_PER_GAME = Math.max(0, config.getInt("Max Actions Per Game"));
                }

                JSONArray participantsArray = config.getJSONArray("Participants");
                int nPlayers = participantsArray.length();
                Run.PlayerType[] playerTypes = new Run.PlayerType[nPlayers];
                Types.TRIBE[] tribes = new Types.TRIBE[nPlayers];
                Participant[] participants = parseParticipants(participantsArray);
                for (int i = 0; i < participants.length; i++) {
                    playerTypes[i] = participants[i].playerType;
                    tribes[i] = participants[i].tribe;
                }
                Types.GAME_MODE gameMode = Run.parseGameModeStr(config.getString("Game Mode"));
                Types.MAP_TYPE mapType = config.has("Map Type")
                        ? Run.parseMapTypeStr(config.getString("Map Type"))
                        : Run.DEFAULT_MAP_TYPE;
                Types.MAP_SIZE mapSize = config.has("Map Size")
                        ? Run.parseMapSizeStr(config.getString("Map Size"))
                        : Run.defaultMapSizeForPlayers(nPlayers);
                int gloryTargetScore = Run.parseGloryTargetScore(config);
                long seed = resolveSeed(config.optLong("Seed", -1));

                //1. Play one game with visuals using the Level Generator:
                if (runMode.equalsIgnoreCase("PlayLG")) {
                    String description = "Seed " + seed + " | " + gameMode + " | " + mapType + " | " + mapSize;
                    play(tribes, seed, playerTypes, participants, gameMode, mapSize, mapType, gloryTargetScore, description);

                //2. Play one game with visuals from a file:
                } else if (runMode.equalsIgnoreCase("PlayFile")) {
                    String levelFile = config.getString("Level File");
                    String description = "Seed " + seed + " | " + gameMode + " | " + levelFile;
                    play(levelFile, seed, playerTypes, participants, gameMode, gloryTargetScore, description);

                //3. Play one game with visuals from a savegame
                } else if (runMode.equalsIgnoreCase("Replay")) {
                    String saveGameFile = config.getString("Replay File Name");
                    load(playerTypes, participants, saveGameFile, seed, "Seed " + seed + " | Replay | " + saveGameFile);
                } else {
                    throw new IllegalArgumentException("Run mode '" + runMode + "' is not recognized.");
                }

            } else {
                throw new IllegalStateException("Could not read 'play.json'.");
            }
        } catch(Exception e) {
            throw new IllegalStateException("Failed to start Play runner.", e);
        }
    }

    private static void play(Types.TRIBE[] tribes, long seed, Run.PlayerType[] playerTypes,
                             Participant[] participants, Types.GAME_MODE gameMode,
                             Types.MAP_SIZE mapSize, Types.MAP_TYPE mapType, int gloryTargetScore,
                             String description)
    {
        KeyController ki = new KeyController(true);
        ActionController ac = new ActionController();

        Game game = _prepareGame(tribes, seed, playerTypes, participants, gameMode, mapSize, mapType, gloryTargetScore);
        game.setRunDescription(description);
        Run.runGame(game, ki, ac);
    }

    private static void play(String levelFile, long seed, Run.PlayerType[] playerTypes,
                             Participant[] participants, Types.GAME_MODE gameMode, int gloryTargetScore,
                             String description)
    {
        KeyController ki = new KeyController(true);
        ActionController ac = new ActionController();

        Game game = _prepareGame(levelFile, seed, playerTypes, participants, gameMode, gloryTargetScore);
        game.setRunDescription(description);
        Run.runGame(game, ki, ac);
    }


    private static void load(Run.PlayerType[] playerTypes, Participant[] participants, String saveGameFile,
                             long seed, String description)
    {
        KeyController ki = new KeyController(true);
        ActionController ac = new ActionController();

        Game game = _loadGame(playerTypes, participants, saveGameFile, seed);
        game.setRunDescription(description);
        Run.runGame(game, ki, ac);
    }


    private static Game _prepareGame(String levelFile, long seed, Run.PlayerType[] playerTypes,
                                     Participant[] participants,
                                     Types.GAME_MODE gameMode, int gloryTargetScore)
    {
        if(Constants.VERBOSE) System.out.println("Seed: " + seed);

        ArrayList<Agent> players = getPlayers(playerTypes, participants, seed);

        Game game = new Game();
        game.init(players, levelFile, seed, gameMode, gloryTargetScore);
        return game;
    }

    private static Game _prepareGame(Types.TRIBE[] tribes, long seed, Run.PlayerType[] playerTypes,
                                     Participant[] participants,
                                     Types.GAME_MODE gameMode, Types.MAP_SIZE mapSize, Types.MAP_TYPE mapType,
                                     int gloryTargetScore)
    {
        if(Constants.VERBOSE) System.out.println("Seed: " + seed);

        ArrayList<Agent> players = getPlayers(playerTypes, participants, seed);

        Game game = new Game();

        game.init(players, seed, tribes, seed, gameMode, mapSize, mapType, gloryTargetScore);

        return game;
    }

    private static ArrayList<Agent> getPlayers(Run.PlayerType[] playerTypes, Participant[] participants, long seed)
    {
        ArrayList<Agent> players = new ArrayList<>();

        ArrayList<Integer> allIds = new ArrayList<>();
        for(int i = 0; i < playerTypes.length; ++i)
            allIds.add(i);

        for(int i = 0; i < playerTypes.length; ++i)
        {
            Agent ag = Run.getAgent(playerTypes[i], seed, participants[i].externalCommand, participants[i].agentFactoryConfig);
            assert ag != null;
            ag.setPlayerIDs(i, allIds);
            players.add(ag);
        }
        return players;
    }

    private static Game _loadGame(Run.PlayerType[] playerTypes, Participant[] participants,
                                  String saveGameFile, long seed)
    {
        ArrayList<Agent> players = getPlayers(playerTypes, participants, seed);

        Game game = new Game();
        game.init(players, saveGameFile);
        return game;
    }

    private static long resolveSeed(long configuredSeed) {
        return configuredSeed == -1 ? System.currentTimeMillis() + new Random().nextInt() : configuredSeed;
    }

    private static Participant[] parseParticipants(JSONArray participantsArray) throws Exception {
        Participant[] parsed = new Participant[participantsArray.length()];
        for (int i = 0; i < participantsArray.length(); i++) {
            JSONObject participantJson = participantsArray.getJSONObject(i);
            Run.PlayerType playerType = Run.parsePlayerTypeStr(participantJson.getString("Type"));
            Types.TRIBE tribe = Run.parseTribeStr(participantJson.getString("Tribe"));
            ArrayList<String> externalCommand = parseParticipantCommand(participantJson);
            Run.AgentFactoryConfig config = Run.AgentFactoryConfig.fromConfig(participantJson);
            parsed[i] = new Participant(playerType, tribe, config, externalCommand);
        }
        return parsed;
    }

    private static ArrayList<String> parseParticipantCommand(JSONObject participantJson) throws Exception {
        if (!participantJson.has("External Command") || participantJson.isNull("External Command")) {
            return null;
        }

        return Run.parseExternalCommandValue(participantJson.get("External Command"));
    }

    private static final class Participant {
        private final Run.PlayerType playerType;
        private final Types.TRIBE tribe;
        private final Run.AgentFactoryConfig agentFactoryConfig;
        private final ArrayList<String> externalCommand;

        private Participant(Run.PlayerType playerType, Types.TRIBE tribe,
                            Run.AgentFactoryConfig agentFactoryConfig, ArrayList<String> externalCommand) {
            this.playerType = playerType;
            this.tribe = tribe;
            this.agentFactoryConfig = agentFactoryConfig;
            this.externalCommand = externalCommand == null ? null : new ArrayList<>(externalCommand);
        }
    }

}
