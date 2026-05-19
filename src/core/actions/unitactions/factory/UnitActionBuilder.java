package core.actions.unitactions.factory;

import core.actions.Action;
import core.actions.ActionFactory;
import core.actors.units.*;
import core.game.GameState;

import java.util.ArrayList;

public class UnitActionBuilder
{
    private static final ActionFactory[] FACTORIES = new ActionFactory[]{
            new AttackFactory(),
            new CaptureFactory(),
            new ConvertFactory(),
            new DisbandFactory(),
            new ExamineFactory(),
            new HealOthersFactory(),
            new InfiltrateFactory(),
            new MakeVeteranFactory(),
            new MoveFactory(),
            new RecoverFactory()
    };

    public ArrayList<Action> getActions(GameState gs, Unit unit)
    {
        ArrayList<Action> allActions = new ArrayList<>();

        if(unit.getTribeId() != gs.getActiveTribeID())
        {
            throw new IllegalStateException("Tried to create unit actions for unit " + unit.getActorId()
                    + " owned by tribe " + unit.getTribeId()
                    + " while active tribe is " + gs.getActiveTribeID() + ".");
        }

        //Upgrade (always possible)
        allActions.addAll(new UpgradeFactory().computeActionVariants(unit, gs));

        if(unit.isFinished())
            return allActions;

        for (ActionFactory factory : FACTORIES) {
            allActions.addAll(factory.computeActionVariants(unit, gs));
        }

        return allActions;
    }

}
