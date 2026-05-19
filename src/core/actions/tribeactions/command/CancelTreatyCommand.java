package core.actions.tribeactions.command;

import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.tribeactions.CancelTreaty;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.GameState;

import java.util.ArrayList;

public class CancelTreatyCommand implements ActionCommand {
    @Override
    public boolean execute(Action a, GameState gs) {
        CancelTreaty action = (CancelTreaty) a;
        if (action.isFeasible(gs)) {
            int tribeId = action.getTribeId();
            int targetId = action.getTargetID();
            Tribe tribe = gs.getTribe(tribeId);

            ArrayList<Integer> unitIds = new ArrayList<>();
            for (int cityId : tribe.getCitiesID()) {
                City city = (City) gs.getActor(cityId);
                if (city != null) {
                    unitIds.addAll(city.getUnitsID());
                }
            }
            unitIds.addAll(tribe.getExtraUnits());

            ArrayList<Unit> unitsToDisband = new ArrayList<>();
            for (Integer unitId : unitIds) {
                Unit unit = (Unit) gs.getActor(unitId);
                if (unit == null) {
                    continue;
                }

                unit.transitionToStatus(core.Types.TURN_STATUS.FINISHED);
                City territoryOwner = gs.getBoard().getCityInBorders(unit.getPosition().x, unit.getPosition().y);
                if (territoryOwner != null && territoryOwner.getTribeId() == targetId) {
                    unitsToDisband.add(unit);
                }
            }

            for (Unit unit : unitsToDisband) {
                gs.killUnit(unit);
            }

            gs.getBoard().getDiplomacy().cancelTreaty(tribeId, targetId, gs.getTick());
            gs.getBoard().recomputeTradeNetworkForTribe(tribeId);
            gs.getBoard().recomputeTradeNetworkForTribe(targetId);
            return true;
        }
        return false;
    }
}
