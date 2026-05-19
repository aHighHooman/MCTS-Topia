package core.actions.unitactions.command;

import core.TechnologyTree;
import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.unitactions.Examine;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.CarrierUnit;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

import static core.Types.EXAMINE_BONUS.UNIT;

public class ExamineCommand implements ActionCommand {

    @Override
    public boolean execute(Action a, GameState gs) {
        Examine action = (Examine)a;
        int unitId = action.getUnitId();

        if(action.isFeasible(gs)) {
            Unit unit = (Unit) gs.getActor(unitId);
            Tribe t = gs.getTribe(unit.getTribeId());
            Random rnd = gs.getRandomGenerator();
            TechnologyTree technologyTree = t.getTechTree();

            int handlerCityId = findRuinHandlerCity(gs, t);

            Types.EXAMINE_BONUS bonus = pickAvailableBonus(gs, unit, t, rnd, technologyTree, handlerCityId);
            if (bonus == null) {
                return false;
            }

            Vector2d unitPos = unit.getPosition();

            switch (bonus) {
                case RESEARCH:
                    boolean researched = technologyTree.researchAtRandom(rnd);
                    if(!researched) {
                        throw new IllegalStateException("Ruin research bonus could not find an available tech at tick "
                                + gs.getTick() + ".");
                    }
                    break;

                case POP_GROWTH:
                    if (handlerCityId != -1) {
                        City c = (City) gs.getActor(handlerCityId);
                        c.addPopulation(t, bonus.getBonus());
                    }
                    break;

                case EXPLORER:
                    gs.getBoard().launchExplorer(unitPos.x, unitPos.y, unit.getTribeId(), rnd);
                    break;

                case RESOURCES:
                    gs.getTribe(unit.getTribeId()).addStars(bonus.getBonus());
                    break;

                case UNIT:
                    spawnRuinUnit(gs, unit, handlerCityId);
                    break;
            }

            gs.getBoard().setResourceAt(unitPos.x, unitPos.y, null);
            if (bonus != UNIT) {
                unit.setStatus(Types.TURN_STATUS.FINISHED);
            }
            return true;
        }
        return false;
    }

    private int findRuinHandlerCity(GameState gs, Tribe tribe) {
        if (tribe.getCitiesID().isEmpty()) {
            return -1;
        }

        int handlerCityId = -1;
        int bestCapitalLevel = Integer.MIN_VALUE;
        for (int cityId : tribe.getCitiesID()) {
            City city = (City) gs.getActor(cityId);
            if (city != null && city.isCapital() && city.getLevel() > bestCapitalLevel) {
                bestCapitalLevel = city.getLevel();
                handlerCityId = cityId;
            }
        }

        if (handlerCityId != -1) {
            return handlerCityId;
        }
        return tribe.controlsCapital() ? tribe.getCapitalID() : tribe.getCitiesID().get(0);
    }

    private Types.EXAMINE_BONUS pickAvailableBonus(GameState gs, Unit unit, Tribe tribe, Random rnd,
                                                   TechnologyTree technologyTree, int handlerCityId) {
        ArrayList<Types.EXAMINE_BONUS> available = new ArrayList<>();
        available.add(Types.EXAMINE_BONUS.RESOURCES);

        if (!technologyTree.isEverythingResearched()) {
            available.add(Types.EXAMINE_BONUS.RESEARCH);
        }
        if (handlerCityId != -1) {
            available.add(Types.EXAMINE_BONUS.POP_GROWTH);
        }
        if (hasUnexploredRuinArea(gs, tribe, unit.getPosition())) {
            available.add(Types.EXAMINE_BONUS.EXPLORER);
        }
        available.add(Types.EXAMINE_BONUS.UNIT);

        return available.isEmpty() ? null : available.get(rnd.nextInt(available.size()));
    }

    private boolean hasUnexploredRuinArea(GameState gs, Tribe tribe, Vector2d position) {
        LinkedList<Vector2d> tiles = position.neighborhood(2, 0, gs.getBoard().getSize());
        tiles.add(position);
        for (Vector2d tile : tiles) {
            if (!tribe.isExplored(tile.x, tile.y)) {
                return true;
            }
        }
        return false;
    }

    private void spawnRuinUnit(GameState gs, Unit examiner, int handlerCityId) {
        Board board = gs.getBoard();
        Vector2d ruinPos = examiner.getPosition().copy();
        boolean waterRuin = board.getTerrainAt(ruinPos.x, ruinPos.y).isWater();

        gs.pushUnit(examiner, ruinPos.x, ruinPos.y);

        Types.UNIT rewardType = waterRuin ? Types.UNIT.RAMMER : Types.UNIT.SWORDMAN;
        int cityId = -1;
        City city = null;
        if (handlerCityId != -1) {
            city = (City) gs.getActor(handlerCityId);
            if (city != null && city.canAddUnit()) {
                cityId = handlerCityId;
            }
        }

        Unit reward = Types.UNIT.createUnit(ruinPos, 0, true, cityId, examiner.getTribeId(), rewardType);
        if (waterRuin && reward instanceof CarrierUnit) {
            ((CarrierUnit) reward).setBaseLandUnit(Types.UNIT.WARRIOR);
        }
        if (waterRuin) {
            reward.setMaxHP(15);
            reward.setCurrentHP(15);
        } else {
            reward.setMaxHP(reward.getMaxHP() + 5);
            reward.setCurrentHP(reward.getMaxHP());
        }

        board.addUnit(city, reward);
        gs.getTribe(examiner.getTribeId()).addScore(rewardType.getPoints());
    }
}
