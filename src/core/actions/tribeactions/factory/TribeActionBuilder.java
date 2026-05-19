package core.actions.tribeactions.factory;

import core.actions.Action;
import core.actions.ActionFactory;
import core.actors.Tribe;
import core.game.GameState;

import java.util.ArrayList;

public class TribeActionBuilder
{
    private static final ActionFactory[] FACTORIES = new ActionFactory[]{
            new BuildRoadFactory(),
            new BuildEmbassyFactory(),
            new ResearchTechFactory(),
            new ProposePeaceFactory(),
            new AcceptPeaceFactory(),
            new ProposeTreatyFactory(),
            new AcceptTreatyFactory(),
            new CancelTreatyFactory(),
            new EndTurnFactory()
    };

    public ArrayList<Action> getActions(GameState gs, Tribe tribe)
    {
        ArrayList<Action> allActions = new ArrayList<>();

        if(tribe.getTribeId() != gs.getActiveTribeID())
        {
            throw new IllegalStateException("Tried to create tribe actions for tribe " + tribe.getTribeId()
                    + " while active tribe is " + gs.getActiveTribeID() + ".");
        }

        for (ActionFactory factory : FACTORIES) {
            allActions.addAll(factory.computeActionVariants(tribe, gs));
        }

        return allActions;
    }

}
