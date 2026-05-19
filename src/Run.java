import core.Types;
import core.game.Game;
import org.json.JSONArray;
import players.*;
import gui.GUI;
import gui.WindowInput;
import players.ExternalProcessAgent;
import org.json.JSONObject;

import java.util.ArrayList;
import java.nio.file.Paths;

import static core.Constants.*;
import static core.Types.TRIBE.*;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

class Run {

    static final Types.MAP_TYPE DEFAULT_MAP_TYPE = Types.MAP_TYPE.CONTINENTS;
    static final int DEFAULT_GLORY_TARGET_SCORE = 10000;

    /**
     * Runs 1 game.
     * @param g - game to run
     * @param ki - Key controller
     * @param ac - Action controller
     */
    static void runGame(Game g, KeyController ki, ActionController ac) {
        WindowInput wi = null;
        GUI frame = null;
        if (VISUALS) {
            wi = new WindowInput();
            wi.windowClosed = false;
            frame = new GUI(g, "Tribes", wi, ac, false);
            frame.addWindowListener(wi);
            frame.addKeyListener(ki);
        }

        g.run(frame, wi);
    }


    /**
     * Runs a game, no visuals nor human player
     * @param g - game to run
     */
    static void runGame(Game g) {
        g.run(null, null);
    }


    public enum PlayerType
    {
        HUMAN,
        EXTERNAL
    }

    static final class AgentFactoryConfig {
        final ExternalProcessAgent.Options externalOptions;

        AgentFactoryConfig(ExternalProcessAgent.Options externalOptions) {
            this.externalOptions = externalOptions == null ? new ExternalProcessAgent.Options() : externalOptions.copy();
        }

        AgentFactoryConfig withExternalOptions(ExternalProcessAgent.Options options) {
            return new AgentFactoryConfig(options);
        }

        static AgentFactoryConfig fromStatics() {
            return new AgentFactoryConfig(new ExternalProcessAgent.Options());
        }

        static AgentFactoryConfig fromConfig(JSONObject config) throws Exception {
            return new AgentFactoryConfig(externalOptionsFromConfig(config));
        }

        private static ExternalProcessAgent.Options externalOptionsFromConfig(JSONObject config) {
            ExternalProcessAgent.Options options = new ExternalProcessAgent.Options();
            if (config == null) {
                return options;
            }

            options.startupTimeoutMillis = getLong(config, "External Startup Timeout Ms", options.startupTimeoutMillis);
            options.actionTimeoutMillis = getLong(config, "External Action Timeout Ms", options.actionTimeoutMillis);
            options.shutdownTimeoutMillis = getLong(config, "External Shutdown Timeout Ms", options.shutdownTimeoutMillis);
            options.matchTimeoutMillis = getLong(config, "External Match Timeout Ms", options.matchTimeoutMillis);

            if (config.has("External Log Dir")) {
                options.logRoot = Paths.get(config.getString("External Log Dir"));
            }
            if (config.has("External Env")) {
                org.json.JSONObject env = config.getJSONObject("External Env");
                for (String key : env.keySet()) {
                    options.environment.put(key, env.getString(key));
                }
            }
            return options;
        }

        private static long getLong(JSONObject config, String key, long defaultValue) {
            return config != null && config.has(key) ? config.getLong(key) : defaultValue;
        }
    }


    static Run.PlayerType parsePlayerTypeStr(String arg) throws Exception
    {
        switch(arg)
        {
            case "Human": return Run.PlayerType.HUMAN;
            case "External": return Run.PlayerType.EXTERNAL;
        }
        throw new Exception("Error: unrecognized Player Type: " + arg);
    }

    static Types.TRIBE parseTribeStr(String arg) throws Exception
    {
        switch(arg)
        {
            case "Xin Xi": return XIN_XI;
            case "Imperius": return IMPERIUS;
            case "Bardur": return BARDUR;
            case "Oumaji": return OUMAJI;
            case "Kickoo": return KICKOO;
            case "Hoodrick": return HOODRICK;
            case "Luxidoor": return LUXIDOOR;
            case "Vengir": return VENGIR;
            case "Zebasi": return ZEBASI;
            case "Ai-Mo": return AI_MO;
            case "Quetzali": return QUETZALI;
            case "Yadakk": return YADAKK;
        }
        throw new Exception("Error: unrecognized Tribe: " + arg);
    }

    static Types.GAME_MODE parseGameModeStr(String arg) throws Exception {
        if (arg == null) {
            return Types.GAME_MODE.PERFECTION;
        }

        switch (arg.trim().toLowerCase()) {
            case "capitals":
                return Types.GAME_MODE.CAPITALS;
            case "might":
                return Types.GAME_MODE.MIGHT;
            case "perfection":
                return Types.GAME_MODE.PERFECTION;
            case "domination":
                return Types.GAME_MODE.DOMINATION;
            case "glory":
                return Types.GAME_MODE.GLORY;
        }
        throw new Exception("Error: unrecognized Game Mode: " + arg);
    }

    static Types.MAP_TYPE parseMapTypeStr(String arg) throws Exception {
        if (arg == null) {
            return DEFAULT_MAP_TYPE;
        }

        switch (arg.trim().toLowerCase()) {
            case "drylands":
                return Types.MAP_TYPE.DRYLANDS;
            case "lakes":
                return Types.MAP_TYPE.LAKES;
            case "continents":
                return Types.MAP_TYPE.CONTINENTS;
            case "pangea":
                return Types.MAP_TYPE.PANGEA;
            case "archipelago":
                return Types.MAP_TYPE.ARCHIPELAGO;
            case "water world":
            case "water_world":
                return Types.MAP_TYPE.WATER_WORLD;
        }
        throw new Exception("Error: unrecognized Map Type: " + arg);
    }

    static Types.MAP_SIZE parseMapSizeStr(String arg) throws Exception {
        if (arg == null) {
            return null;
        }

        switch (arg.trim().toLowerCase()) {
            case "tiny":
            case "121":
                return Types.MAP_SIZE.TINY;
            case "small":
            case "196":
                return Types.MAP_SIZE.SMALL;
            case "normal":
            case "256":
                return Types.MAP_SIZE.NORMAL;
            case "large":
            case "324":
                return Types.MAP_SIZE.LARGE;
            case "huge":
            case "400":
                return Types.MAP_SIZE.HUGE;
            case "massive":
            case "900":
                return Types.MAP_SIZE.MASSIVE;
        }
        throw new Exception("Error: unrecognized Map Size: " + arg);
    }

    static Types.MAP_SIZE defaultMapSizeForPlayers(int nPlayers) {
        if (nPlayers <= 2) return Types.MAP_SIZE.TINY;
        if (nPlayers == 3) return Types.MAP_SIZE.SMALL;
        if (nPlayers == 4) return Types.MAP_SIZE.NORMAL;
        if (nPlayers == 5) return Types.MAP_SIZE.LARGE;
        if (nPlayers == 6) return Types.MAP_SIZE.HUGE;
        return Types.MAP_SIZE.MASSIVE;
    }

    static int parseGloryTargetScore(JSONObject config) throws Exception {
        if (config == null || !config.has("Glory Score Limit")) {
            return DEFAULT_GLORY_TARGET_SCORE;
        }

        int value = config.getInt("Glory Score Limit");
        switch (value) {
            case 5000:
            case 10000:
            case 15000:
            case 20000:
            case 25000:
                return value;
            default:
                throw new Exception("Error: unsupported Glory Score Limit: " + value);
        }
    }

    public static ArrayList<ArrayList<String>> parseExternalCommands(JSONObject config, int nPlayers) throws Exception
    {
        ArrayList<ArrayList<String>> commands = new ArrayList<>();
        for (int i = 0; i < nPlayers; i++) {
            commands.add(null);
        }

        if (config == null || !config.has("External Commands")) {
            return commands;
        }

        JSONArray externalCommands = config.getJSONArray("External Commands");
        if (externalCommands.length() != nPlayers) {
            throw new Exception("External Commands must match the number of players");
        }

        for (int i = 0; i < externalCommands.length(); i++) {
            if (externalCommands.isNull(i)) {
                continue;
            }

            Object rawCommand = externalCommands.get(i);
            commands.set(i, parseExternalCommandValue(rawCommand));
        }

        return commands;
    }

    public static ArrayList<String> parseExternalCommandValue(Object rawCommand) throws Exception
    {
        ArrayList<String> command = new ArrayList<>();
        if (rawCommand instanceof JSONArray) {
            JSONArray commandArray = (JSONArray) rawCommand;
            for (int i = 0; i < commandArray.length(); i++) {
                command.add(commandArray.getString(i));
            }
        } else if (rawCommand instanceof String) {
            Pattern tokenPattern = Pattern.compile("\"([^\"]*)\"|'([^']*)'|(\\S+)");
            Matcher matcher = tokenPattern.matcher((String) rawCommand);
            while (matcher.find()) {
                if (matcher.group(1) != null) {
                    command.add(matcher.group(1));
                } else if (matcher.group(2) != null) {
                    command.add(matcher.group(2));
                } else {
                    command.add(matcher.group(3));
                }
            }
        } else {
            throw new Exception("Unsupported external command format: " + rawCommand);
        }

        if (command.isEmpty()) {
            throw new Exception("External command cannot be empty");
        }
        return command;
    }

    public static Agent getAgent(Run.PlayerType playerType, long agentSeed, ArrayList<String> externalCommand)
    {
        return getAgent(playerType, agentSeed, externalCommand, AgentFactoryConfig.fromStatics());
    }

    public static Agent getAgent(Run.PlayerType playerType, long agentSeed, ArrayList<String> externalCommand,
                                 AgentFactoryConfig config)
    {
        AgentFactoryConfig effectiveConfig = config == null ? AgentFactoryConfig.fromStatics() : config;
        switch (playerType)
        {
            case EXTERNAL:
                if (externalCommand == null || externalCommand.isEmpty()) {
                    throw new IllegalArgumentException("External player selected without an External Commands entry.");
                }
                return new ExternalProcessAgent(agentSeed, externalCommand, effectiveConfig.externalOptions);
        }
        return null;
    }
}
