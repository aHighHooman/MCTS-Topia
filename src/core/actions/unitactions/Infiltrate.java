package core.actions.unitactions;

import core.Types;
import core.actions.Action;
import core.actors.City;
import core.actors.units.Unit;
import core.game.GameState;
import utils.Vector2d;

public class Infiltrate extends UnitAction {

    private int targetCityId;

    public Infiltrate(int unitId) {
        super(Types.ACTION.INFILTRATE);
        this.unitId = unitId;
    }

    public void setTargetCityId(int targetCityId) {
        this.targetCityId = targetCityId;
    }

    public int getTargetCityId() {
        return targetCityId;
    }

    @Override
    public boolean isFeasible(final GameState gs) {
        Unit unit = (Unit) gs.getActor(unitId);
        City targetCity = (City) gs.getActor(targetCityId);

        if (unit == null || targetCity == null ||
                (unit.getType() != Types.UNIT.CLOAK && unit.getType() != Types.UNIT.DINGHY) ||
                !unit.canAttack()) {
            return false;
        }

        if (targetCity.isInfiltrated()) {
            return false;
        }

        if (targetCity.getTribeId() == unit.getTribeId() ||
                gs.getBoard().getDiplomacy().isTreaty(unit.getTribeId(), targetCity.getTribeId())) {
            return false;
        }

        Vector2d cityPos = targetCity.getPosition();
        Unit cityOccupant = gs.getBoard().getUnitAt(cityPos.x, cityPos.y);
        if (cityOccupant != null && cityOccupant.getTribeId() != targetCity.getTribeId()) {
            return false;
        }

        if (unit.getStatus() == Types.TURN_STATUS.MOVED && !unit.wasHiddenAtTurnStart()) {
            return false;
        }

        if (!gs.getTribe(unit.getTribeId()).isExplored(cityPos.x, cityPos.y)) {
            return false;
        }

        return Vector2d.chebychevDistance(unit.getPosition(), cityPos) <= unit.RANGE;
    }

    @Override
    public Action copy() {
        Infiltrate infiltrate = new Infiltrate(unitId);
        infiltrate.setTargetCityId(targetCityId);
        return infiltrate;
    }
}
