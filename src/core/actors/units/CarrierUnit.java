package core.actors.units;

import core.Types;
import utils.Vector2d;

public abstract class CarrierUnit extends Unit {

    private Types.UNIT baseLandUnit;

    protected CarrierUnit(double atk, double def, int mov, int maxHp, int range, int cost,
                          Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(atk, def, mov, maxHp, range, cost, pos, kills, isVeteran, cityId, tribeId);
    }

    public Types.UNIT getBaseLandUnit() {
        return baseLandUnit;
    }

    public void setBaseLandUnit(Types.UNIT baseLandUnit) {
        this.baseLandUnit = baseLandUnit;
    }

    protected void copyCarrierTo(CarrierUnit copy) {
        copyBaseTo(copy);
        copy.setBaseLandUnit(getBaseLandUnit());
    }
}
