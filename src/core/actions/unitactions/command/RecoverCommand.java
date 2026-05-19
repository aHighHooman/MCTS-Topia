package core.actions.unitactions.command;

import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.unitactions.Recover;
import core.actors.City;
import core.actors.units.Unit;
import core.game.GameState;

import static core.TribesConfig.RECOVER_IN_BORDERS_PLUS_HP;
import static core.TribesConfig.RECOVER_PLUS_HP;

public class RecoverCommand implements ActionCommand {

    @Override
    public boolean execute(Action a, GameState gs) {
        Recover action = (Recover)a;
        int unitId = action.getUnitId();

        Unit unit = (Unit) gs.getActor(unitId);
        if(unit == null)
            return false;

        double currentHP = unit.getCurrentHPExact();
        int addHP = RECOVER_PLUS_HP;

        //Check if action is feasible before execution
        if (action.isFeasible(gs)) {

            City territoryOwner = gs.getBoard().getCityInBorders(unit.getPosition().x, unit.getPosition().y);
            if (territoryOwner != null && territoryOwner.getTribeId() == unit.getTribeId()) {
                addHP += RECOVER_IN_BORDERS_PLUS_HP;
            }
            unit.setCurrentHP(Math.min(currentHP + addHP, unit.getMaxHP()));
            unit.transitionToStatus(Types.TURN_STATUS.FINISHED);
            return true;
        }
        return false;
    }
}
