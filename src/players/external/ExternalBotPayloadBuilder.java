package players.external;

import core.TechnologyTree;
import core.Types;
import core.actions.Action;
import core.actions.cityactions.Build;
import core.actions.cityactions.CityAction;
import core.actions.cityactions.LevelUp;
import core.actions.cityactions.ResourceGathering;
import core.actions.cityactions.Spawn;
import core.actions.tribeactions.AcceptPeace;
import core.actions.tribeactions.AcceptTreaty;
import core.actions.tribeactions.BuildEmbassy;
import core.actions.tribeactions.BuildRoad;
import core.actions.tribeactions.CancelTreaty;
import core.actions.tribeactions.ProposePeace;
import core.actions.tribeactions.ProposeTreaty;
import core.actions.tribeactions.ResearchTech;
import core.actions.tribeactions.TribeAction;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Capture;
import core.actions.unitactions.Convert;
import core.actions.unitactions.Examine;
import core.actions.unitactions.Infiltrate;
import core.actions.unitactions.Move;
import core.actions.unitactions.UnitAction;
import core.actors.Building;
import core.actors.City;
import core.actors.Temple;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import core.game.TribeResult;
import org.json.JSONArray;
import org.json.JSONObject;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.TreeSet;

public final class ExternalBotPayloadBuilder {

    public static final String ROOT_STATE_ID = "root";
    private static JSONObject extraResultFields = null;

    private ExternalBotPayloadBuilder() {
    }

    public static JSONObject buildActionRequest(GameState gs, int playerId, long remainingTimeMs, int requestId,
                                                ArrayList<Action> legalActions, ArrayList<String> actionIds,
                                                String rootStateId, boolean forwardModelEnabled) {
        JSONObject payload = new JSONObject();
        payload.put("type", "action_request");
        payload.put("request_id", requestId);
        payload.put("player_id", playerId);
        payload.put("time_ms", remainingTimeMs);
        payload.put("obs", serializeObservation(gs, playerId));
        payload.put("actions", serializeActions(legalActions, actionIds));
        if (forwardModelEnabled) {
            payload.put("fm", buildForwardModelDescriptor(rootStateId));
        }
        return payload;
    }

    public static JSONObject buildResult(GameState gs, int playerId, double reward) {
        JSONObject payload = new JSONObject();
        payload.put("type", "game_over");
        payload.put("player_id", playerId);
        payload.put("reward", reward);
        payload.put("winner", winnerId(gs));
        payload.put("scores", finalScores(gs));
        payload.put("rank", serializeRanking(gs.getCurrentRanking()));
        payload.put("term", normalizedTerminalReward(gs, playerId));
        if (extraResultFields != null) {
            for (String key : extraResultFields.keySet()) {
                payload.put(key, extraResultFields.get(key));
            }
        }
        return payload;
    }

    public static void setExtraResultFields(JSONObject fields) {
        extraResultFields = fields == null ? null : new JSONObject(fields.toString());
    }

    public static JSONObject buildForwardModelDescriptor(String rootStateId) {
        JSONObject payload = new JSONObject();
        payload.put("root", rootStateId);
        payload.put("cmd", new JSONArray()
                .put("inspect")
                .put("step")
                .put("step_many")
                .put("release"));
        return payload;
    }

    public static JSONObject buildForwardModelStatePayload(GameState gs, int playerId,
                                                           ArrayList<Action> legalActions) {
        return buildForwardModelStatePayload(gs, playerId, legalActions, null);
    }

    public static JSONObject buildForwardModelStatePayload(GameState gs, int playerId,
                                                           ArrayList<Action> legalActions,
                                                           ArrayList<String> actionIds) {
        JSONObject payload = new JSONObject();
        payload.put("obs", serializeObservation(gs, playerId));
        payload.put("actions", serializeActions(legalActions, actionIds));
        payload.put("terminal", gs.isGameOver() || legalActions.isEmpty());
        payload.put("active", gs.getActiveTribeID());
        if (gs.isGameOver()) {
            payload.put("winner", winnerId(gs));
            payload.put("scores", finalScores(gs));
            payload.put("rank", serializeRanking(gs.getCurrentRanking()));
        }
        return payload;
    }

    public static JSONArray serializeActions(ArrayList<Action> legalActions) {
        return serializeActions(legalActions, null);
    }

    public static JSONArray serializeActions(ArrayList<Action> legalActions, ArrayList<String> actionIds) {
        JSONArray actions = new JSONArray();
        for (int i = 0; i < legalActions.size(); i++) {
            JSONObject action = serializeAction(legalActions.get(i), i);
            if (actionIds != null && i < actionIds.size() && actionIds.get(i) != null) {
                action.put("id", actionIds.get(i));
            } else {
                action.put("id", "A" + i);
            }
            actions.put(action);
        }
        return actions;
    }

    private static JSONObject serializeObservation(GameState gs, int playerId) {
        GameState observed = gs.copyForPlayer(playerId);
        JSONObject observation = new JSONObject();
        observation.put("tick", observed.getTick());
        observation.put("mode", observed.getGameMode().name());
        observation.put("map", observed.getMapType() == null ? JSONObject.NULL : observed.getMapType().name());
        observation.put("active", observed.getActiveTribeID());
        observation.put("end", observed.canEndTurn(observed.getActiveTribeID()));
        observation.put("lvlup", observed.isLevelingUp());
        observation.put("tribes", serializeTribes(observed));
        observation.put("cities", serializeCities(observed));
        observation.put("units", serializeUnits(observed));
        observation.put("board", serializeBoard(observed, playerId));
        observation.put("rank", serializeRanking(observed.getCurrentRanking()));
        observation.put("rel", serializeRelationships(observed, playerId));
        observation.put("capitals", serializeCapitalCities(gs));
        return observation;
    }

    private static JSONArray serializeCapitalCities(GameState gs) {
        JSONArray capitals = new JSONArray();
        for (Tribe tribe : gs.getTribes()) {
            int capitalId = tribe.getCapitalID();
            if (capitalId <= 0) {
                continue;
            }
            City city = (City) gs.getActor(capitalId);
            if (city == null) {
                continue;
            }
            JSONObject out = new JSONObject();
            out.put("id", city.getActorId());
            out.put("p", city.getTribeId());
            out.put("x", city.getPosition().x);
            out.put("y", city.getPosition().y);
            out.put("lvl", city.getLevel());
            out.put("pop", city.getPopulation());
            out.put("need", city.getPopulation_need());
            out.put("prod", city.getProduction());
            out.put("cap", true);
            capitals.put(out);
        }
        return capitals;
    }

    private static JSONArray serializeTribes(GameState gs) {
        JSONArray tribes = new JSONArray();
        for (Tribe tribe : gs.getTribes()) {
            JSONObject out = new JSONObject();
            out.put("id", tribe.getTribeId());
            out.put("tribe", tribe.getType().name());
            out.put("stars", tribe.getStars());
            out.put("score", tribe.getScore());
            out.put("res", tribe.getWinner().name());
            out.put("cap", tribe.getCapitalID());
            out.put("tech", researchedTechIds(tribe.getTechTree()));
            out.put("cities", serializeIntegerList(tribe.getCitiesID()));
            out.put("extra", serializeIntegerList(tribe.getExtraUnits()));
            out.put("conn", serializeIntegerList(tribe.getConnectedCities()));
            out.put("met", serializeIntegerList(tribe.getTribesMet()));
            out.put("known_caps", serializeIntegerList(tribe.getKnownCapitalTribes()));
            out.put("lights", serializeIntegerList(tribe.getDiscoveredLighthouses()));
            out.put("kills", tribe.getnKills());
            out.put("pacifist", tribe.getnPacifistCount());
            out.put("disabled", tribe.isUnitsDisabledNextTurn());
            out.put("mon", serializeMonuments(tribe));
            tribes.put(out);
        }
        return tribes;
    }

    private static JSONObject serializeMonuments(Tribe tribe) {
        JSONObject out = new JSONObject();
        for (Types.BUILDING building : tribe.getMonuments().keySet()) {
            out.put(building.name(), tribe.getMonuments().get(building).name());
        }
        return out;
    }

    private static JSONArray researchedTechIds(TechnologyTree tree) {
        return tree == null ? new JSONArray() : tree.getResearchedTechIds();
    }

    private static JSONArray serializeCities(GameState gs) {
        JSONArray cities = new JSONArray();
        for (Tribe tribe : gs.getTribes()) {
            for (Integer cityId : tribe.getCitiesID()) {
                City city = (City) gs.getActor(cityId);
                if (city == null) {
                    continue;
                }

                JSONObject out = new JSONObject();
                out.put("id", city.getActorId());
                out.put("p", city.getTribeId());
                out.put("x", city.getPosition().x);
                out.put("y", city.getPosition().y);
                out.put("lvl", city.getLevel());
                out.put("pop", city.getPopulation());
                out.put("need", city.getPopulation_need());
                out.put("prod", city.getProduction());
                out.put("cap", city.isCapital());
                out.put("wall", city.hasWalls());
                out.put("bound", city.getBound());
                out.put("pts", city.getPointsWorth());
                out.put("inf", city.isInfiltrated());
                out.put("units", serializeIntegerList(city.getUnitsID()));
                out.put("b", serializeBuildings(city.getBuildings()));
                cities.put(out);
            }
        }
        return cities;
    }

    private static JSONArray serializeBuildings(Iterable<Building> buildings) {
        JSONArray out = new JSONArray();
        for (Building building : buildings) {
            JSONObject entry = new JSONObject();
            entry.put("t", building.type.name());
            entry.put("x", building.position.x);
            entry.put("y", building.position.y);
            entry.put("city", building.cityId);
            entry.put("owner", building.getStoredOwnerTribeId());
            if (building instanceof Temple) {
                Temple temple = (Temple) building;
                entry.put("lvl", temple.getLevel());
                entry.put("score_turns", temple.getTurnsToScore());
            }
            out.put(entry);
        }
        return out;
    }

    private static JSONArray serializeUnits(GameState gs) {
        JSONArray units = new JSONArray();
        for (Tribe tribe : gs.getTribes()) {
            for (Unit unit : gs.getUnits(tribe.getTribeId())) {
                if (unit == null) {
                    continue;
                }

                JSONObject out = new JSONObject();
                out.put("id", unit.getActorId());
                out.put("p", unit.getTribeId());
                out.put("c", unit.getCityId());
                out.put("t", unit.getType().name());
                out.put("x", unit.getPosition().x);
                out.put("y", unit.getPosition().y);
                out.put("hp", unit.getCurrentHP());
                out.put("hpx", unit.getCurrentHPExact());
                out.put("mhp", unit.getMaxHP());
                out.put("k", unit.getKills());
                out.put("v", unit.isVeteran());
                out.put("s", unit.getStatus().name());
                out.put("h", unit.isHidden());
                out.put("hts", unit.wasHiddenAtTurnStart());
                if (unit.hasHiddenEnemyHint()) {
                    out.put("hint", true);
                }
                out.put("atk", unit.getAttackValue());
                out.put("def", unit.getDefenceValue());
                out.put("mov", unit.MOV);
                out.put("r", unit.RANGE);
                out.put("cost", unit.COST);
                units.put(out);
            }
        }
        return units;
    }

    private static JSONObject serializeBoard(GameState gs, int playerId) {
        Board board = gs.getBoard();
        Tribe observer = gs.getTribe(playerId);
        JSONObject out = new JSONObject();
        out.put("size", board.getSize());
        out.put("terrain", buildTileMatrix(board, observer, TileField.TERRAIN));
        out.put("resource", buildTileMatrix(board, observer, TileField.RESOURCE));
        out.put("building", buildTileMatrix(board, observer, TileField.BUILDING));
        out.put("city", buildTileMatrix(board, observer, TileField.CITY));
        out.put("unit", buildTileMatrix(board, observer, TileField.UNIT));
        out.put("exp", buildTileMatrix(board, observer, TileField.EXPLORED));
        out.put("road", buildTileMatrix(board, observer, TileField.ROAD));
        out.put("lighthouses", serializeLighthouses(board, observer));
        return out;
    }

    private static JSONObject serializeLighthouses(Board board, Tribe observer) {
        JSONObject out = new JSONObject();
        int max = board.getSize() - 1;
        int[][] corners = new int[][]{
                {0, 0},
                {0, max},
                {max, 0},
                {max, max}
        };

        for (int i = 0; i < corners.length; i++) {
            int x = corners[i][0];
            int y = corners[i][1];
            String key = Integer.toString(i);
            if (!observer.isExplored(x, y) || board.getResourceAt(x, y) != Types.RESOURCE.LIGHTHOUSE) {
                out.put(key, JSONObject.NULL);
                continue;
            }

            JSONObject lighthouse = new JSONObject();
            lighthouse.put("x", x);
            lighthouse.put("y", y);
            JSONArray discoverers = new JSONArray();
            for (Integer tribeId : board.getLighthouseDiscoverers(x, y)) {
                discoverers.put(tribeId);
            }
            lighthouse.put("seen", discoverers);
            out.put(key, lighthouse);
        }
        return out;
    }

    private static JSONArray buildTileMatrix(Board board, Tribe observer, TileField field) {
        JSONArray rows = new JSONArray();
        int size = board.getSize();
        for (int y = 0; y < size; y++) {
            JSONArray row = new JSONArray();
            for (int x = 0; x < size; x++) {
                boolean explored = observer.isExplored(x, y);
                switch (field) {
                    case TERRAIN:
                        row.put(explored ? stringOrNull(board.getTerrainAt(x, y)) : JSONObject.NULL);
                        break;
                    case RESOURCE:
                        row.put(explored ? stringOrNull(board.getResourceAt(x, y)) : JSONObject.NULL);
                        break;
                    case BUILDING:
                        row.put(explored ? stringOrNull(board.getBuildingAt(x, y)) : JSONObject.NULL);
                        break;
                    case CITY:
                        row.put(explored ? board.getCityIdAt(x, y) : 0);
                        break;
                    case UNIT:
                        row.put(explored ? unitId(board, x, y) : 0);
                        break;
                    case EXPLORED:
                        row.put(explored ? 1 : 0);
                        break;
                    case ROAD:
                        row.put(explored && board.checkTradeNetwork(x, y) ? 1 : 0);
                        break;
                }
            }
            rows.put(row);
        }
        return rows;
    }

    private static JSONArray serializeIntegerList(Iterable<Integer> values) {
        JSONArray out = new JSONArray();
        for (Integer value : values) {
            out.put(value);
        }
        return out;
    }

    private static int unitId(Board board, int x, int y) {
        Unit unit = board.getUnitAt(x, y);
        return unit == null ? 0 : unit.getActorId();
    }

    private static JSONArray serializeRanking(TreeSet<TribeResult> ranking) {
        JSONArray out = new JSONArray();
        for (TribeResult tribeResult : ranking) {
            out.put(tribeResult.getId());
        }
        return out;
    }

    private static JSONArray finalScores(GameState gs) {
        JSONArray out = new JSONArray();
        for (Tribe tribe : gs.getTribes()) {
            JSONObject entry = new JSONObject();
            entry.put("id", tribe.getTribeId());
            entry.put("score", tribe.getScore());
            entry.put("res", tribe.getWinner().name());
            out.put(entry);
        }
        return out;
    }

    private static Object winnerId(GameState gs) {
        for (Tribe tribe : gs.getTribes()) {
            if (tribe.getWinner() == Types.RESULT.WIN) {
                return tribe.getTribeId();
            }
        }
        return JSONObject.NULL;
    }

    private static double normalizedTerminalReward(GameState gs, int playerId) {
        Object winner = winnerId(gs);
        if (winner == JSONObject.NULL) {
            return 0.0;
        }
        int winnerInt = ((Number) winner).intValue();
        return winnerInt == playerId ? 1.0 : -1.0;
    }

    private static JSONArray serializeRelationships(GameState gs, int playerId) {
        JSONArray rows = new JSONArray();
        Types.RELATIONSHIP[][] relationships = gs.getVisibleRelationships(playerId);
        for (Types.RELATIONSHIP[] relationshipRow : relationships) {
            JSONArray row = new JSONArray();
            for (Types.RELATIONSHIP relationship : relationshipRow) {
                row.put(relationship == null ? JSONObject.NULL : relationship.name());
            }
            rows.put(row);
        }
        return rows;
    }

    private static JSONObject serializeAction(Action action, int index) {
        JSONObject out = new JSONObject();
        out.put("i", index);
        out.put("t", action.getActionType().name());

        if (action instanceof UnitAction) {
            out.put("u", ((UnitAction) action).getUnitId());
        }
        if (action instanceof CityAction) {
            CityAction cityAction = (CityAction) action;
            out.put("c", cityAction.getCityId());
            putXY(out, cityAction.getTargetPos());
        }
        if (action instanceof TribeAction) {
            out.put("p", ((TribeAction) action).getTribeId());
        }

        if (action instanceof Move) {
            putXY(out, ((Move) action).getDestination());
        } else if (action instanceof Attack) {
            out.put("tu", ((Attack) action).getTargetId());
        } else if (action instanceof Convert) {
            out.put("tu", ((Convert) action).getTargetId());
        } else if (action instanceof Infiltrate) {
            out.put("tc", ((Infiltrate) action).getTargetCityId());
        } else if (action instanceof Examine) {
            Types.EXAMINE_BONUS bonus = ((Examine) action).getBonus();
            if (bonus != null) {
                out.put("bonus", bonus.name());
            }
        } else if (action instanceof Capture) {
            Capture capture = (Capture) action;
            out.put("tc", capture.getTargetCity());
            out.put("ct", capture.getCaptureType().name());
        } else if (action instanceof Spawn) {
            out.put("ut", ((Spawn) action).getUnitType().name());
        } else if (action instanceof Build) {
            out.put("bt", ((Build) action).getBuildingType().name());
        } else if (action instanceof ResourceGathering) {
            out.put("rt", ((ResourceGathering) action).getResource().name());
        } else if (action instanceof LevelUp) {
            out.put("b", ((LevelUp) action).getBonus().name());
        } else if (action instanceof ResearchTech) {
            out.put("tech", ((ResearchTech) action).getTech().name());
        } else if (action instanceof BuildRoad) {
            putXY(out, ((BuildRoad) action).getPosition());
        } else if (action instanceof BuildEmbassy) {
            out.put("tp", ((BuildEmbassy) action).getTargetID());
        } else if (action instanceof ProposePeace) {
            out.put("tp", ((ProposePeace) action).getTargetID());
        } else if (action instanceof ProposeTreaty) {
            out.put("tp", ((ProposeTreaty) action).getTargetID());
        } else if (action instanceof AcceptPeace) {
            out.put("tp", ((AcceptPeace) action).getTargetID());
        } else if (action instanceof AcceptTreaty) {
            out.put("tp", ((AcceptTreaty) action).getTargetID());
        } else if (action instanceof CancelTreaty) {
            out.put("tp", ((CancelTreaty) action).getTargetID());
        }

        return out;
    }

    private static void putXY(JSONObject out, Vector2d position) {
        if (position != null) {
            out.put("x", position.x);
            out.put("y", position.y);
        }
    }

    private static Object stringOrNull(Enum<?> value) {
        return value == null ? JSONObject.NULL : value.name();
    }

    private enum TileField {
        TERRAIN,
        RESOURCE,
        BUILDING,
        CITY,
        UNIT,
        EXPLORED,
        ROAD
    }
}
