package core.actors.units;

import core.Types;
import utils.Vector2d;

import static core.TribesConfig.*;

public class Pirate extends CarrierUnit {

    public Pirate(Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(PIRATE_ATTACK, PIRATE_DEFENCE, PIRATE_MOVEMENT, PIRATE_MAX_HP, PIRATE_RANGE, PIRATE_COST,
                pos, kills, isVeteran, cityId, tribeId);
    }

    @Override
    public Types.UNIT getType() {
        return Types.UNIT.PIRATE;
    }

    @Override
    public Pirate copy(boolean hideInfo) {
        Pirate c = new Pirate(getPosition(), getKills(), isVeteran(), getCityId(), getTribeId());
        copyCarrierTo(c);
        return hideInfo ? (Pirate) c.hide() : c;
    }
}
