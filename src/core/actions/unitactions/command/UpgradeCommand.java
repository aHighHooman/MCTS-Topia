package core.actions.unitactions.command;

import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.unitactions.Upgrade;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.CarrierUnit;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;

import static core.Types.UNIT.*;
public class UpgradeCommand implements ActionCommand {

    @Override
    public boolean execute(Action a, GameState gs) {
        Upgrade action = (Upgrade)a;
        int unitId = action.getUnitId();

        Unit unit = (Unit) gs.getActor(unitId);
        Tribe tribe = gs.getTribe(unit.getTribeId());
        Board board = gs.getBoard();
        City city = (City) board.getActor(unit.getCityId());

        if(action.isFeasible(gs)){
            Types.UNIT unitType = unit.getType();
            Types.UNIT nextType;

            //get the correct type - or nothing!
            switch (action.getActionType()) {
                case UPGRADE_RAMMER:
                    nextType = RAMMER;
                    break;
                case UPGRADE_SCOUT:
                    nextType = SCOUT;
                    break;
                case UPGRADE_BOMBER:
                    nextType = BOMBER;
                    break;
                default:
                    return false;
            }

            if (unitType != RAFT && !(unitType == SCOUT && action.getActionType() == Types.ACTION.UPGRADE_BOMBER)) {
                return false;
            }

            //Create the new unit
            Unit newUnit = Types.UNIT.createUnit(unit.getPosition(), unit.getKills(), unit.isVeteran(), unit.getCityId(), unit.getTribeId(), nextType);
            newUnit.setCurrentHP(unit.getCurrentHP());
            newUnit.setMaxHP(unit.getMaxHP());
            Types.UNIT baseLandUnit = unit instanceof CarrierUnit ? ((CarrierUnit) unit).getBaseLandUnit() : null;
            if (newUnit instanceof CarrierUnit) {
                ((CarrierUnit) newUnit).setBaseLandUnit(baseLandUnit);
            }

            //adjustments in tribe and board.
            tribe.subtractStars(nextType.getCost());

            Types.TURN_STATUS turn_status = unit.getStatus();
            //We first remove the unit, so there's space for the new one to take its place.
            board.removeUnitFromBoard(unit);
            board.removeUnitFromCity(unit, city, tribe);
            board.addUnit(city, newUnit);
            newUnit.setStatus(turn_status);
            return true;
        }
        return false;
    }
}

