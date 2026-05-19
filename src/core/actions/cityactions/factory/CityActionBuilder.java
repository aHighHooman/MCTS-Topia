package core.actions.cityactions.factory;

import core.actions.Action;
import core.actions.ActionFactory;
import core.actors.City;
import core.game.GameState;

import java.util.ArrayList;

public class CityActionBuilder
{
    private static final ActionFactory LEVEL_UP_FACTORY = new LevelUpFactory();
    private static final ActionFactory[] FACTORIES = new ActionFactory[]{
            new BuildFactory(),
            new BurnForestFactory(),
            new ClearForestFactory(),
            new DestroyFactory(),
            new GrowForestFactory(),
            new ResourceGatheringFactory(),
            new SpawnFactory()
    };

    private boolean levelUpFlag;

    public CityActionBuilder()
    {
        levelUpFlag = false;
    }

    public ArrayList<Action> getActions(GameState gs, City city)
    {
        levelUpFlag = false;
        ArrayList<Action> allActions = new ArrayList<>();

        if(city.getTribeId() != gs.getActiveTribeID())
        {
            throw new IllegalStateException("Tried to create city actions for city " + city.getActorId()
                    + " while active tribe is " + gs.getActiveTribeID() + ".");
        }


        //Level Up
        allActions.addAll(LEVEL_UP_FACTORY.computeActionVariants(city, gs));

        if(!allActions.isEmpty())
        {
            //Level up is special. Nothing else can be done here
            levelUpFlag = true;
            return allActions;
        }

        for (ActionFactory factory : FACTORIES) {
            allActions.addAll(factory.computeActionVariants(city, gs));
        }

        return allActions;
    }


    public boolean cityLevelsUp() {
        return levelUpFlag;
    }
}
