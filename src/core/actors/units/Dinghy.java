package core.actors.units;

import core.Types;
import utils.Vector2d;

import static core.TribesConfig.*;

public class Dinghy extends CarrierUnit {

    public Dinghy(Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(DINGHY_ATTACK, DINGHY_DEFENCE, DINGHY_MOVEMENT, CLOAK_MAX_HP, DINGHY_RANGE, DINGHY_COST,
                pos, kills, isVeteran, cityId, tribeId);
    }

    @Override
    public Types.UNIT getType() {
        return Types.UNIT.DINGHY;
    }

    @Override
    public Dinghy copy(boolean hideInfo) {
        Dinghy c = new Dinghy(getPosition(), getKills(), isVeteran(), getCityId(), getTribeId());
        copyCarrierTo(c);
        return hideInfo ? (Dinghy) c.hide() : c;
    }
}
