package core.actors.units;

import core.Types;
import utils.Vector2d;

import static core.TribesConfig.*;

public class Juggernaut extends CarrierUnit {

    public Juggernaut(Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(JUGGERNAUT_ATTACK, JUGGERNAUT_DEFENCE, JUGGERNAUT_MOVEMENT, JUGGERNAUT_MAX_HP,
                JUGGERNAUT_RANGE, JUGGERNAUT_COST, pos, kills, isVeteran, cityId, tribeId);
    }

    @Override
    public Types.UNIT getType() {
        return Types.UNIT.JUGGERNAUT;
    }

    @Override
    public Juggernaut copy(boolean hideInfo) {
        Juggernaut c = new Juggernaut(getPosition(), getKills(), isVeteran(), getCityId(), getTribeId());
        copyCarrierTo(c);
        return hideInfo ? (Juggernaut) c.hide() : c;
    }
}
