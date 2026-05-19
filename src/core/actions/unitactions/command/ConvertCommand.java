package core.actions.unitactions.command;

import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.unitactions.Convert;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.GameState;

public class ConvertCommand implements ActionCommand {

    @Override
    public boolean execute(Action a, GameState gs) {
        Convert action = (Convert) a;

        //Check if action is feasible before execution
        if (action.isFeasible(gs)) {
            int unitId = action.getUnitId();
            int targetId = action.getTargetId();
            Unit target = (Unit) gs.getActor(targetId);
            Unit unit = (Unit) gs.getActor(unitId);
            int targetTribeId = target.getTribeId();
            Tribe targetTribe = gs.getTribe(targetTribeId);

            //remove the unit from its original city.
            int cityId = target.getCityId();
            City c = (City) gs.getActor(cityId);
            gs.getBoard().removeUnitFromCity(target, c, targetTribe);

            //add tribe to converted unit
            target.setTribeId(unit.getTribeId());
            gs.getActiveTribe().addExtraUnit(target);

            gs.getBoard().getDiplomacy().clearPendingOffersBetween(unit.getTribeId(), targetTribeId);
            gs.getBoard().getDiplomacy().setRelationship(unit.getTribeId(), targetTribeId, Types.RELATIONSHIP.WAR, gs.getTick());
            gs.getBoard().destroyEmbassiesBetween(gs, unit.getTribeId(), targetTribeId);

            //manage status of the units after the action is executed
            unit.transitionToStatus(Types.TURN_STATUS.ATTACKED);
            target.setStatus(Types.TURN_STATUS.FINISHED);
            return true;
        }
        return false;
    }
}
