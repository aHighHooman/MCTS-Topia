package core.actions.unitactions;

import core.TribesConfig;
import core.actions.Action;
import core.game.GameState;
import core.actors.units.Unit;
import core.Types;

public class MakeVeteran extends UnitAction
{
    public MakeVeteran(int unitId)
    {
        super(Types.ACTION.MAKE_VETERAN);
        super.unitId = unitId;
    }

    @Override
    public boolean isFeasible(final GameState gs) {
        Unit unit = (Unit) gs.getActor(this.unitId);
        if(unit.getType() == Types.UNIT.SUPERUNIT || unit.getType().isWaterUnit() ||
                unit.getType() == Types.UNIT.CLOAK || unit.getType() == Types.UNIT.DAGGER)
            return false;
        return unit.getKills() >= TribesConfig.VETERAN_KILLS && !unit.isVeteran();
    }


    @Override
    public Action copy() {
        return new MakeVeteran(this.unitId);
    }

    public String toString() {
        return "MAKE_VETERAN by unit " + this.unitId;
    }
}
