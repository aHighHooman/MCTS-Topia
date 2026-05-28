package core.game;

import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import org.json.JSONArray;
import org.json.JSONObject;
import players.external.ExternalBotPayloadBuilder;
import players.external.ExternalForwardModelSession;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

public final class NativeParityOracle {

    public static final int PROTOCOL_VERSION = 2;

    private NativeParityOracle() {
    }

    public static void main(String[] args) {
        try {
            String fixture = requiredFlag(args, "--fixture");
            int playerId = intFlag(args, "--player", -1);
            int depth = Math.max(1, intFlag(args, "--depth", 1));
            int maxStates = Math.max(1, intFlag(args, "--max-states", 24));
            int maxActionsPerState = Math.max(1, intFlag(args, "--max-actions-per-state", 8));
            boolean pretty = booleanFlag(args, "--pretty");

            GameState root = loadFixture(fixture);
            if (!root.isGameOver() && root.getActiveTribe() != null) {
                root.computePlayerActions(root.getActiveTribe());
            }
            if (playerId < 0) {
                playerId = root.getActiveTribeID();
            }

            ArrayList<Action> actions = root.getAllAvailableActions();
            ArrayList<String> actionIds = buildActionIds("A", actions.size());
            ExternalForwardModelSession session = new ExternalForwardModelSession(root, playerId, actions, actionIds);

            JSONObject out = new JSONObject();
            out.put("protocol_version", PROTOCOL_VERSION);
            out.put("fixture", fixture);
            out.put("player_id", playerId);
            out.put("root_state_id", session.getRootStateId());
            out.put("root", inspectRoot(session));
            JSONArray nodes = buildParityTree(session, depth, maxStates, maxActionsPerState);
            out.put("nodes", nodes);
            out.put("children", nodes.getJSONObject(0).getJSONArray("children"));

            System.out.println(pretty ? out.toString(2) : out.toString());
        } catch (Exception e) {
            JSONObject error = new JSONObject();
            error.put("protocol_version", PROTOCOL_VERSION);
            error.put("ok", false);
            error.put("error", e.getClass().getName() + ": " + e.getMessage());
            System.out.println(error.toString());
            System.exit(1);
        }
    }

    private static GameState loadFixture(String fixture) {
        if ("smoke".equals(fixture) || "smoke:basic".equals(fixture)) {
            return buildSmokeFixture();
        }
        if ("milestone3:city".equals(fixture)) {
            return buildMilestone3CityFixture();
        }
        if ("milestone3:level-up".equals(fixture)) {
            return buildMilestone3LevelUpFixture();
        }
        if ("milestone3:tribe-war".equals(fixture)) {
            return buildMilestone3TribeFixture(Types.RELATIONSHIP.WAR, false, 0);
        }
        if ("milestone3:tribe-peace".equals(fixture)) {
            return buildMilestone3TribeFixture(Types.RELATIONSHIP.PEACE, false, 0);
        }
        if ("milestone3:accept-peace".equals(fixture)) {
            return buildMilestone3TribeFixture(Types.RELATIONSHIP.WAR, true, 1);
        }
        if ("milestone3:accept-treaty".equals(fixture)) {
            return buildMilestone3TribeFixture(Types.RELATIONSHIP.PEACE, true, 1);
        }
        if ("milestone3:cancel-treaty".equals(fixture)) {
            return buildMilestone3TribeFixture(Types.RELATIONSHIP.TREATY, false, 0);
        }
        if ("milestone4:units".equals(fixture)) {
            return buildMilestone4UnitFixture();
        }
        GameLoader loader = new GameLoader(fixture);
        GameState state = new GameState(
                new Random(loader.getSeed()),
                loader.getGame_mode(),
                loader.getGloryTargetScore(),
                loader.getTribes(),
                loader.getBoard(),
                loader.getTick());
        state.setGloryResolutionRound(loader.getGloryResolutionRound());
        state.setGameIsOver(loader.getGameIsOver());
        return state;
    }

    private static GameState buildSmokeFixture() {
        Random random = new Random(1337L);
        Tribe[] tribes = new Tribe[]{
                new Tribe(Types.TRIBE.IMPERIUS),
                new Tribe(Types.TRIBE.BARDUR)
        };
        Board board = new Board();
        board.init(5, tribes);
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                board.setTerrainAt(x, y, Types.TERRAIN.PLAIN);
            }
        }

        City city0 = addCapital(board, tribes, random, 0, 1, 1);
        addCapital(board, tribes, random, 1, 3, 3);
        Unit warrior = Types.UNIT.createUnit(new Vector2d(1, 2), 0, false,
                city0.getActorId(), 0, Types.UNIT.WARRIOR);
        board.addUnit(city0, warrior);
        tribes[0].addScore(Types.UNIT.WARRIOR.getPoints());
        tribes[0].addStars(5);
        tribes[1].addStars(5);
        board.setActiveTribeID(0);

        GameState state = new GameState(random, Types.GAME_MODE.MIGHT,
                GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE, tribes, board, 0);
        state.setMapType(Types.MAP_TYPE.CONTINENTS);
        board.revealExplorationFromCurrentAssets(random);
        return state;
    }

    private static GameState buildMilestone3CityFixture() {
        Random random = new Random(20240301L);
        Tribe[] tribes = new Tribe[]{
                new Tribe(Types.TRIBE.IMPERIUS),
                new Tribe(Types.TRIBE.BARDUR)
        };
        Board board = new Board();
        board.init(7, tribes);
        fillTerrain(board, 7, Types.TERRAIN.PLAIN);

        City city = addCapital(board, tribes, random, 0, 3, 3);
        addCapital(board, tribes, random, 1, 6, 6);
        board.setTerrainAt(2, 3, Types.TERRAIN.FOREST);
        board.setTerrainAt(3, 2, Types.TERRAIN.PLAIN);
        board.setTerrainAt(4, 3, Types.TERRAIN.PLAIN);
        board.setTerrainAt(3, 4, Types.TERRAIN.PLAIN);
        board.setResourceAt(3, 2, Types.RESOURCE.FRUIT);
        board.setBuildingAt(3, 4, Types.BUILDING.FARM);

        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.FORESTRY);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.FARMING);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.CONSTRUCTION);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.SPIRITUALISM);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.CHIVALRY);
        tribes[0].setStars(50);
        board.setActiveTribeID(0);

        GameState state = new GameState(random, Types.GAME_MODE.MIGHT,
                GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE, tribes, board, 0);
        state.setMapType(Types.MAP_TYPE.CONTINENTS);
        board.assignCityTiles(city, city.getBound());
        board.revealExplorationFromCurrentAssets(random);
        return state;
    }

    private static GameState buildMilestone3LevelUpFixture() {
        Random random = new Random(20240302L);
        Tribe[] tribes = new Tribe[]{
                new Tribe(Types.TRIBE.IMPERIUS),
                new Tribe(Types.TRIBE.BARDUR)
        };
        Board board = new Board();
        board.init(5, tribes);
        fillTerrain(board, 5, Types.TERRAIN.PLAIN);

        City city = addCapital(board, tribes, random, 0, 2, 2);
        addCapital(board, tribes, random, 1, 4, 4);
        city.setPopulation(city.getPopulation_need());
        city.levelUp();
        city.setPopulation(city.getPopulation_need());
        tribes[0].setStars(10);
        board.setActiveTribeID(0);

        GameState state = new GameState(random, Types.GAME_MODE.MIGHT,
                GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE, tribes, board, 0);
        state.setMapType(Types.MAP_TYPE.CONTINENTS);
        board.revealExplorationFromCurrentAssets(random);
        return state;
    }

    private static GameState buildMilestone3TribeFixture(Types.RELATIONSHIP relationship,
                                                         boolean pendingOffer,
                                                         int activeTribeId) {
        Random random = new Random(20240303L + activeTribeId + relationship.getKey());
        Tribe[] tribes = new Tribe[]{
                new Tribe(Types.TRIBE.IMPERIUS),
                new Tribe(Types.TRIBE.BARDUR)
        };
        Board board = new Board();
        board.init(7, tribes);
        fillTerrain(board, 7, Types.TERRAIN.PLAIN);

        addCapital(board, tribes, random, 0, 1, 1);
        addCapital(board, tribes, random, 1, 5, 5);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.STRATEGY);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        tribes[1].getTechTree().doResearchInit(Types.TECHNOLOGY.ROADS);
        tribes[1].getTechTree().doResearchInit(Types.TECHNOLOGY.STRATEGY);
        tribes[1].getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        tribes[0].getTribesMet().add(1);
        tribes[1].getTribesMet().add(0);
        tribes[0].setStars(7);
        tribes[1].setStars(7);
        board.getDiplomacy().setRelationship(0, 1, relationship, 0);
        if (pendingOffer) {
            board.getDiplomacy().setPendingTreatyOffer(0, 1, 0);
            board.getDiplomacy().setPendingOfferType(0, 1, relationship == Types.RELATIONSHIP.WAR
                    ? Types.RELATIONSHIP.PEACE
                    : Types.RELATIONSHIP.TREATY);
        }
        board.setActiveTribeID(activeTribeId);

        GameState state = new GameState(random, Types.GAME_MODE.MIGHT,
                GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE, tribes, board, 0);
        state.setMapType(Types.MAP_TYPE.CONTINENTS);
        board.revealExplorationFromCurrentAssets(random);
        return state;
    }

    private static GameState buildMilestone4UnitFixture() {
        Random random = new Random(20240401L);
        Tribe[] tribes = new Tribe[]{
                new Tribe(Types.TRIBE.IMPERIUS),
                new Tribe(Types.TRIBE.BARDUR)
        };
        Board board = new Board();
        board.init(9, tribes);
        fillTerrain(board, 9, Types.TERRAIN.PLAIN);

        City home = addCapital(board, tribes, random, 0, 1, 1);
        City enemy = addCapital(board, tribes, random, 1, 7, 7);
        board.setTerrainAt(3, 1, Types.TERRAIN.VILLAGE);

        Unit warrior = addUnit(board, home, Types.UNIT.WARRIOR, 2, 2);
        warrior.setStatus(Types.TURN_STATUS.FRESH);
        warrior.setKills(TribesConfig.VETERAN_KILLS);
        Unit defender = addUnit(board, enemy, Types.UNIT.WARRIOR, 2, 3);
        defender.setStatus(Types.TURN_STATUS.FRESH);
        Unit capturer = addUnit(board, home, Types.UNIT.WARRIOR, 3, 1);
        capturer.setStatus(Types.TURN_STATUS.FRESH);
        Unit mindBender = addUnit(board, home, Types.UNIT.MIND_BENDER, 5, 5);
        mindBender.setStatus(Types.TURN_STATUS.FRESH);
        Unit convertTarget = addUnit(board, enemy, Types.UNIT.WARRIOR, 5, 6);
        convertTarget.setStatus(Types.TURN_STATUS.FRESH);
        Unit wounded = addUnit(board, home, Types.UNIT.ARCHER, 1, 2);
        wounded.setCurrentHP(4);
        wounded.setStatus(Types.TURN_STATUS.FRESH);
        Unit raft = addUnit(board, home, Types.UNIT.RAFT, 6, 1);
        raft.setStatus(Types.TURN_STATUS.FRESH);

        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.FREE_SPIRIT);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.RAMMING);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.SAILING);
        tribes[0].getTechTree().doResearchInit(Types.TECHNOLOGY.NAVIGATION);
        tribes[0].setStars(50);
        tribes[1].setStars(10);
        board.setActiveTribeID(0);

        GameState state = new GameState(random, Types.GAME_MODE.MIGHT,
                GameState.RunDefaults.DEFAULT_GLORY_TARGET_SCORE, tribes, board, 0);
        state.setMapType(Types.MAP_TYPE.CONTINENTS);
        board.revealExplorationFromCurrentAssets(random);
        return state;
    }

    private static void fillTerrain(Board board, int size, Types.TERRAIN terrain) {
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                board.setTerrainAt(x, y, terrain);
            }
        }
    }

    private static City addCapital(Board board, Tribe[] tribes, Random random, int tribeId, int x, int y) {
        City city = new City(x, y, tribeId);
        city.setCapital(true);
        board.setTerrainAt(x, y, Types.TERRAIN.CITY);
        board.addCityToTribe(city, random);
        board.assignCityTiles(city, city.getBound());
        tribes[tribeId].addScore(TribesConfig.CITY_CENTRE_POINTS);
        return city;
    }

    private static Unit addUnit(Board board, City city, Types.UNIT unitType, int x, int y) {
        Unit unit = unitType.createUnit(new Vector2d(x, y), 0, false,
                -1, city.getTribeId(), unitType);
        board.addUnit(city, unit);
        return unit;
    }

    private static JSONObject inspectRoot(ExternalForwardModelSession session) {
        JSONObject request = new JSONObject();
        request.put("command", "inspect");
        request.put("state_id", session.getRootStateId());
        JSONObject response = session.handleCommand(request);
        if (!"forward_model_result".equals(response.optString("type"))) {
            throw new IllegalStateException(response.optString("message", "Oracle root inspect failed."));
        }
        return response.getJSONObject("state");
    }

    private static JSONArray buildParityTree(ExternalForwardModelSession session,
                                             int maxDepth,
                                             int maxStates,
                                             int maxActionsPerState) {
        JSONArray nodes = new JSONArray();
        ArrayList<String> queue = new ArrayList<>();
        ArrayList<Integer> queueDepths = new ArrayList<>();
        queue.add(session.getRootStateId());
        queueDepths.add(0);

        int cursor = 0;
        while (cursor < queue.size() && nodes.length() < maxStates) {
            String stateId = queue.get(cursor);
            int depth = queueDepths.get(cursor);
            cursor++;

            JSONObject state = inspectState(session, stateId);
            JSONArray children = stepSampledActions(session, stateId, state, maxActionsPerState);

            JSONObject node = new JSONObject();
            node.put("state_id", stateId);
            node.put("depth", depth);
            node.put("state", state);
            node.put("children", children);
            nodes.put(node);

            if (depth + 1 >= maxDepth) {
                continue;
            }
            for (int i = 0; i < children.length() && queue.size() < maxStates; i++) {
                JSONObject child = children.getJSONObject(i);
                if (!child.optBoolean("ok", false)) {
                    continue;
                }
                JSONObject childState = child.optJSONObject("state");
                if (childState == null || childState.optBoolean("is_terminal", false)) {
                    continue;
                }
                String childStateId = childState.optString("state_id", "");
                if (!childStateId.isEmpty()) {
                    queue.add(childStateId);
                    queueDepths.add(depth + 1);
                }
            }
        }
        return nodes;
    }

    private static JSONObject inspectState(ExternalForwardModelSession session, String stateId) {
        JSONObject request = new JSONObject();
        request.put("command", "inspect");
        request.put("state_id", stateId);
        JSONObject response = session.handleCommand(request);
        if (!"forward_model_result".equals(response.optString("type"))) {
            throw new IllegalStateException(response.optString("message", "Oracle inspect failed for " + stateId));
        }
        return response.getJSONObject("state");
    }

    private static JSONArray stepSampledActions(ExternalForwardModelSession session,
                                                String stateId,
                                                JSONObject state,
                                                int maxActionsPerState) {
        JSONArray children = new JSONArray();
        JSONArray actions = state.optJSONArray("actions");
        int actionCount = actions == null ? 0 : actions.length();
        int limit = Math.min(actionCount, maxActionsPerState);
        for (int i = 0; i < limit; i++) {
            JSONObject child = new JSONObject();
            child.put("source_state_id", stateId);
            child.put("action_index", i);
            child.put("action_id", actionIdFor(stateId, i));
            try {
                JSONObject request = new JSONObject();
                request.put("command", "step");
                request.put("state_id", stateId);
                request.put("action_index", i);
                JSONObject response = session.handleCommand(request);
                child.put("ok", "forward_model_result".equals(response.optString("type")));
                if (child.getBoolean("ok")) {
                    child.put("state", response.getJSONObject("state"));
                } else {
                    child.put("error", response.optString("message", "Oracle step failed."));
                }
            } catch (Exception e) {
                child.put("ok", false);
                child.put("error", e.getClass().getName() + ": " + e.getMessage());
            }
            children.put(child);
        }
        return children;
    }

    private static String actionIdFor(String stateId, int actionIndex) {
        String prefix = ExternalBotPayloadBuilder.ROOT_STATE_ID.equals(stateId) ? "A" : stateId + "_A";
        return prefix + actionIndex;
    }

    private static ArrayList<String> buildActionIds(String prefix, int size) {
        ArrayList<String> ids = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            ids.add(prefix + i);
        }
        return ids;
    }

    private static String requiredFlag(String[] args, String name) {
        String value = flagValue(args, name, null);
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException("Missing required flag " + name);
        }
        return value;
    }

    private static int intFlag(String[] args, String name, int fallback) {
        String value = flagValue(args, name, null);
        return value == null ? fallback : Integer.parseInt(value);
    }

    private static boolean booleanFlag(String[] args, String name) {
        for (String arg : args) {
            if (name.equals(arg)) {
                return true;
            }
        }
        return false;
    }

    private static String flagValue(String[] args, String name, String fallback) {
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.equals(name) && i + 1 < args.length) {
                return args[i + 1];
            }
            String prefix = name + "=";
            if (arg.startsWith(prefix)) {
                return arg.substring(prefix.length());
            }
        }
        return fallback;
    }
}
