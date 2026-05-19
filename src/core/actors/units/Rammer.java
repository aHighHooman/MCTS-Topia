package core.actors.units;

import core.Types;
import utils.Vector2d;

import static core.TribesConfig.*;

public class Rammer extends CarrierUnit {

    public Rammer(Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(RAMMER_ATTACK, RAMMER_DEFENCE, RAMMER_MOVEMENT, -1, RAMMER_RANGE, RAMMER_COST,
                pos, kills, isVeteran, cityId, tribeId);
    }

    @Override
    public Types.UNIT getType() {
        return Types.UNIT.RAMMER;
    }

    @Override
    public Rammer copy(boolean hideInfo) {
        Rammer c = new Rammer(getPosition(), getKills(), isVeteran(), getCityId(), getTribeId());
        copyCarrierTo(c);
        return hideInfo ? (Rammer) c.hide() : c;
    }
}
