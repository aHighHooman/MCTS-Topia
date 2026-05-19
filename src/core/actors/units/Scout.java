package core.actors.units;

import core.Types;
import utils.Vector2d;

import static core.TribesConfig.*;

public class Scout extends CarrierUnit
{
    public Scout(Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(SCOUT_ATTACK, SCOUT_DEFENCE, SCOUT_MOVEMENT, -1, SCOUT_RANGE, SCOUT_COST, pos, kills, isVeteran, cityId, tribeId);
    }

    @Override
    public Types.UNIT getType() {
        return Types.UNIT.SCOUT;
    }

    @Override
    public Scout copy(boolean hideInfo) {
        Scout c = new Scout(getPosition(), getKills(), isVeteran(), getCityId(), getTribeId());
        copyCarrierTo(c);
        return hideInfo ? (Scout) c.hide() : c;
    }
}

