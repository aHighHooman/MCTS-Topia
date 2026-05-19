package core.actions.tribeactions;

import core.Diplomacy;
import core.Types;
import core.UnlockRules;
import core.actions.Action;
import core.actors.Tribe;
import core.game.GameState;

public class ProposePeace extends TribeAction {

    private int targetID;

    public ProposePeace(int tribeId) {
        super(Types.ACTION.PROPOSE_PEACE);
        this.tribeId = tribeId;
    }

    public void setTargetID(int targetID) { this.targetID = targetID; }
    public int getTargetID() { return this.targetID; }

    @Override
    public boolean isFeasible(final GameState gs) {
        Diplomacy diplomacy = gs.getBoard().getDiplomacy();
        Tribe tribe = gs.getTribe(tribeId);
        return tribeId != targetID &&
                UnlockRules.isActionUnlocked(tribe, Types.ACTION.PROPOSE_PEACE) &&
                diplomacy.isAtWar(tribeId, targetID) &&
                !diplomacy.hasPendingOfferBetween(tribeId, targetID);
    }

    @Override
    public Action copy() {
        ProposePeace proposePeace = new ProposePeace(this.tribeId);
        proposePeace.setTargetID(targetID);
        return proposePeace;
    }

    public String toString() {
        return "PROPOSE_PEACE by tribe " + this.tribeId + " to tribe " + this.targetID;
    }
}
