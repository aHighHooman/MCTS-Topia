package core.actors.units;

import core.Types;
import utils.Vector2d;

import static core.TribesConfig.*;

public class Cloak extends Unit {
    public Cloak(Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(CLOAK_ATTACK, CLOAK_DEFENCE, CLOAK_MOVEMENT, CLOAK_MAX_HP, CLOAK_RANGE, CLOAK_COST, pos, kills, isVeteran, cityId, tribeId);
    }

    @Override
    public Types.UNIT getType() {
        return Types.UNIT.CLOAK;
    }

    @Override
    public Cloak copy(boolean hideInfo) {
        Cloak c = new Cloak(getPosition(), getKills(), isVeteran(), getCityId(), getTribeId());
        copyBaseTo(c);
        return hideInfo ? (Cloak) c.hide() : c;
    }
}
