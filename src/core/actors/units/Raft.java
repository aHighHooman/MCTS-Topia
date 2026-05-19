package core.actors.units;

import core.Types;
import utils.Vector2d;

import static core.TribesConfig.*;

public class Raft extends CarrierUnit
{
    public Raft(Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(RAFT_ATTACK, RAFT_DEFENCE, RAFT_MOVEMENT, -1, RAFT_RANGE, RAFT_COST, pos, kills, isVeteran, cityId, tribeId);
    }

    @Override
    public Types.UNIT getType() {
        return Types.UNIT.RAFT;
    }

    @Override
    public Raft copy(boolean hideInfo) {
        Raft c = new Raft(getPosition(), getKills(), isVeteran(), getCityId(), getTribeId());
        copyCarrierTo(c);
        return hideInfo ? (Raft) c.hide() : c;
    }
}

