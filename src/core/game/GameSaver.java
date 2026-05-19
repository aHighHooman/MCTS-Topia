package core.game;

import core.Types;
import core.actors.Building;
import core.actors.City;
import core.actors.Temple;
import core.actors.Tribe;
import core.actors.units.CarrierUnit;
import core.actors.units.Unit;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

class GameSaver {


    static void writeTurnFile(GameState gs, Board gameBoard, long seed) {
        writeTurnFile(gs, gameBoard, seed, false);
    }

    static void writeTurnFile(GameState gs, Board gameBoard, long seed, boolean pendingTickAdvance) {
        try{
            File rootFileLoc = new File("save/" + seed);
            File turnFile = new File(rootFileLoc, gs.getTick() + "_" + gs.getBoard().getActiveTribeID());

            if (!rootFileLoc.exists()) {
                rootFileLoc.mkdirs();
            }

            turnFile.mkdir();

            // JSON
            JSONObject game = new JSONObject();

            // Board INFO (2D array) - Terrain, Resource, UnitID, CityID, NetWorks
            JSONObject board = new JSONObject();
            JSONArray terrain2D = new JSONArray();
            JSONArray terrain;

            JSONArray resource2D = new JSONArray();
            JSONArray resource;

            JSONArray unit2D = new JSONArray();
            JSONArray units;

            JSONArray city2D = new JSONArray();
            JSONArray cities;

            JSONArray building2D = new JSONArray();
            JSONArray JBuildings;

            JSONArray network2D = new JSONArray();
            JSONArray networks;

            // Unit INFO: id:{type, x, y, kills, isVeteran, cityId, tribeId, HP}
            JSONObject unit = new JSONObject();

            // City INFO: id:{x, y, tribeId, population_need, bound, level, isCapital, population,
            //                production, hasWalls, pointsWorth, building(array)}
            JSONObject city = new JSONObject();

            // Building INFO: {x, y, type, level(optional), turnsToScore(optional), bonus}
            JSONObject building = new JSONObject();

            for (int i=0; i<gameBoard.getSize(); i++){

                // Initial JSON Object for each row
                terrain = new JSONArray();
                resource = new JSONArray();
                units = new JSONArray();
                cities = new JSONArray();
                networks = new JSONArray();
                JBuildings = new JSONArray();

                for(int j=0; j<gameBoard.getSize(); j++){
                    // Save Terrain INFO
                    terrain.put(gs.getBoard().getTerrainAt(i, j).getKey());
                    // Save Resource INFO
                    resource.put(gs.getBoard().getResourceAt(i, j) != null? gs.getBoard().getResourceAt(i, j).getKey():-1);

                    // Save unit INFO
                    units.put(gs.getBoard().getUnitIDAt(i, j));

                    // Save cityTile INFO
                    cities.put(gs.getBoard().getCityIdAt(i, j));

                    // Save Building INFO
                    JBuildings.put(gs.getBoard().getBuildingAt(i, j)!= null? gs.getBoard().getBuildingAt(i, j).getKey():-1);

                    // Save network INFO
                    networks.put(gs.getBoard().getNetworkTilesAt(i, j));

                }
                // Update row value
                terrain2D.put(terrain);
                resource2D.put(resource);
                unit2D.put(units);
                city2D.put(cities);
                network2D.put(networks);
                building2D.put(JBuildings);
            }

            ArrayList<Unit> unitList = getAllUnits(gameBoard);
            for(Unit u: unitList){
                JSONObject uInfo = new JSONObject();
                uInfo.put("type", u.getType().getKey());
                if (u instanceof CarrierUnit){
                    Types.UNIT baseLandType = ((CarrierUnit) u).getBaseLandUnit();
                    if (baseLandType != null) {
                        uInfo.put("baseLandType", baseLandType.getKey());
                    }
                }
                uInfo.put("x", u.getPosition().x);
                uInfo.put("y", u.getPosition().y);
                uInfo.put("kill", u.getKills());
                uInfo.put("isVeteran", u.isVeteran());
                uInfo.put("cityID", u.getCityId());
                uInfo.put("tribeId", u.getTribeId());
                uInfo.put("currentHP", u.getCurrentHPExact());
                uInfo.put("hidden", u.isHidden());
                uInfo.put("hiddenAtTurnStart", u.wasHiddenAtTurnStart());
                uInfo.put("status", u.getStatus().getKey());
                unit.put(String.valueOf(u.getActorId()), uInfo);
            }

            ArrayList<City> citiesList = getAllCities(gameBoard);
            for(City c: citiesList){
                // City INFO: id:{x, y, tribeId, population_need, bound, level, isCapital, population,
                //                production, hasWalls, pointsWorth, building(array)}
                JSONObject cInfo = new JSONObject();
                cInfo.put("x", c.getPosition().x);
                cInfo.put("y", c.getPosition().y);
                cInfo.put("tribeID", c.getTribeId());
                cInfo.put("population_need", c.getPopulation_need());
                cInfo.put("bound", c.getBound());
                cInfo.put("level", c.getLevel());
                cInfo.put("isCapital", c.isCapital());
                cInfo.put("population", c.getPopulation());
                cInfo.put("production", c.getProduction());
                cInfo.put("hasWalls", c.hasWalls());
                cInfo.put("pointsWorth", c.getPointsWorth());
                cInfo.put("infiltrated", c.isInfiltrated());
                // Save Buildings INFO
                JSONArray buildingList = new JSONArray();
                LinkedList<Building> buildings = c.getBuildings();
                if (buildings != null) {
                    for (Building b : buildings) {
                        JSONObject bInfo = new JSONObject();
                        bInfo.put("x", b.position.x);
                        bInfo.put("y", b.position.y);
                        bInfo.put("type", b.type.getKey());
                        bInfo.put("ownerTribeId", b.getOwnerTribeId(c.getTribeId()));
                        if (b.type.isTemple()) {
                            Temple t = (Temple) b;
                            bInfo.put("level", t.getLevel());
                            bInfo.put("turnsToScore", t.getTurnsToScore());
                        }
                        buildingList.put(bInfo);
                    }
                }
                cInfo.put("buildings", buildingList);
                cInfo.put("units", c.getUnitsID());
                city.put(String.valueOf(c.getActorId()), cInfo);
            }

            board.put("terrain", terrain2D);
            board.put("resource", resource2D);
            board.put("unitID", unit2D);
            board.put("cityID", city2D);
            board.put("network", network2D);
            board.put("building", building2D);
            board.put("actorIDcounter", gameBoard.getActorIDcounter());
            board.put("diplomacy", gameBoard.getDiplomacy().toJSON());

            game.put("board", board);
            game.put("unit", unit);
            game.put("city", city);

            // Save Tribes Information () - id: {citiesID, capitalID, type, techTree, stars, winner, score, obsGrid,
            //                                   connectedCities, monuments:{type: status}, tribesMet, extraUnits,
            //                                   nKills, nPacifistCount}
            JSONObject tribesINFO = new JSONObject();
            Tribe[] tribes = gs.getTribes();
            for(Tribe t: tribes){
                JSONObject tribeInfo = new JSONObject();
                tribeInfo.put("citiesID", t.getCitiesID());
                tribeInfo.put("capitalID", t.getCapitalID());
                tribeInfo.put("type", t.getType().getKey());
                JSONObject techINFO = new JSONObject();
                techINFO.put("researched", t.getTechTree().getResearched());
                techINFO.put("researchedTechIds", t.getTechTree().getResearchedTechIds());
                techINFO.put("everythingResearched", t.getTechTree().isEverythingResearched());
                tribeInfo.put("technology", techINFO);
                tribeInfo.put("star", t.getStars());
                tribeInfo.put("winner", t.getWinner().getKey());
                tribeInfo.put("score", t.getScore());
                tribeInfo.put("obsGrid", t.getObsGrid());
                tribeInfo.put("connectedCities", t.getConnectedCities());
                HashMap<Types.BUILDING, Types.BUILDING.MONUMENT_STATUS> m = t.getMonuments();
                JSONObject monumentInfo = new JSONObject();
                for (Types.BUILDING key : m.keySet()) {
                    monumentInfo.put(String.valueOf(key.getKey()), m.get(key).getKey());
                }
                tribeInfo.put("monuments", monumentInfo);
                JSONArray tribesMetInfo = new JSONArray();
                ArrayList<Integer> tribesMet= t.getTribesMet();
                for (Integer tribeId : tribesMet){
                    tribesMetInfo.put(tribeId);
                }
                tribeInfo.put("tribesMet", tribesMetInfo);
                tribeInfo.put("knownCapitalTribes", t.getKnownCapitalTribes());
                tribeInfo.put("discoveredLighthouses", t.getDiscoveredLighthouses());
                tribeInfo.put("extraUnits", t.getExtraUnits());
                tribeInfo.put("nKills", t.getnKills());
                tribeInfo.put("nPacifistCount", t.getnPacifistCount());
                tribeInfo.put("unitsDisabledNextTurn", t.isUnitsDisabledNextTurn());
                tribesINFO.put(String.valueOf(t.getActorId()), tribeInfo);
            }

            game.put("tribes", tribesINFO);
            game.put("schemaVersion", 3);
            game.put("seed", seed);
            game.put("tick", gs.getTick());
            game.put("gameIsOver", gs.isGameOver());
            game.put("activeTribeID", gs.getActiveTribeID());
            game.put("pendingTickAdvance", pendingTickAdvance);
            game.put("gameMode", gs.getGameMode().getKey());
            game.put("gloryTargetScore", gs.getGloryTargetScore());
            game.put("gloryResolutionRound", gs.getGloryResolutionRound());

            FileWriter fw_game = new FileWriter(turnFile.getPath() + "/game.json");
            fw_game.write(game.toString(4));
            fw_game.close();

        } catch (IOException e){
            throw new IllegalStateException("Failed to write save data for seed " + seed + ".", e);
        }
    }

    private static ArrayList<City> getAllCities(Board board)
    {
        Tribe[] tribes = board.getTribes();
        ArrayList<City> cityActors = new ArrayList<>();
        for(Tribe t: tribes){
            ArrayList<Integer> cities = t.getCitiesID();
            for(Integer cityId : cities)
            {
                cityActors.add((City)board.getActor(cityId));
            }
        }

        return cityActors;
    }

    private static ArrayList<Unit> getAllUnits(Board board)
    {
        Tribe[] tribes = board.getTribes();
        ArrayList<Unit> unitActors = new ArrayList<>();
        for(Tribe t: tribes){
            ArrayList<Integer> cities = t.getCitiesID();
            for(Integer cityId : cities)
            {
                City c = (City)board.getActor(cityId);
                for(Integer unitId : c.getUnitsID())
                {
                    Unit unit = (Unit) board.getActor(unitId);
                    unitActors.add(unit);
                }
            }

            for(Integer unitId : t.getExtraUnits())
            {
                Unit unit = (Unit) board.getActor(unitId);
                unitActors.add(unit);
            }
        }

        return unitActors;
    }
}
