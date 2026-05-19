package core.actions.tribeactions;

import core.Diplomacy;
import core.Types;
import core.UnlockRules;
import core.actions.Action;
import core.actors.Tribe;
import core.game.GameState;

public class ProposeTreaty extends TribeAction {

    private int targetID;

    public ProposeTreaty(int tribeId) {
        super(Types.ACTION.PROPOSE_TREATY);
        this.tribeId = tribeId;
    }

    public void setTargetID(int targetID) { this.targetID = targetID; }
    public int getTargetID() { return this.targetID; }

    @Override
    public boolean isFeasible(final GameState gs) {
        Diplomacy diplomacy = gs.getBoard().getDiplomacy();
        Tribe tribe = gs.getTribe(tribeId);
        return tribeId != targetID &&
                UnlockRules.isActionUnlocked(tribe, Types.ACTION.PROPOSE_TREATY) &&
                diplomacy.getRelationship(tribeId, targetID) == Types.RELATIONSHIP.PEACE &&
                !diplomacy.hasPendingOfferBetween(tribeId, targetID);
    }

    @Override
    public Action copy() {
        ProposeTreaty proposeTreaty = new ProposeTreaty(this.tribeId);
        proposeTreaty.setTargetID(targetID);
        return proposeTreaty;
    }

    public String toString() {
        return "PROPOSE_TREATY by tribe " + this.tribeId + " to tribe " + this.targetID;
    }
}
