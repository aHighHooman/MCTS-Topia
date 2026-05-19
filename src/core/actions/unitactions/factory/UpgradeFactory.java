package core.actions.unitactions.factory;

import core.actions.Action;
import core.actions.ActionFactory;
import core.actions.unitactions.Upgrade;
import core.actors.Actor;
import core.actors.units.Unit;
import core.game.GameState;
import core.Types;

import java.util.LinkedList;

public class UpgradeFactory implements ActionFactory {

    @Override
    public LinkedList<Action> computeActionVariants(final Actor actor, final GameState gs) {
        Unit unit = (Unit) actor;
        LinkedList<Action> upgradeActions = new LinkedList<>();

        if (unit.getType() == Types.UNIT.RAFT) {
            addUpgradeIfFeasible(upgradeActions, unit, gs, Types.ACTION.UPGRADE_RAMMER);
            addUpgradeIfFeasible(upgradeActions, unit, gs, Types.ACTION.UPGRADE_SCOUT);
            addUpgradeIfFeasible(upgradeActions, unit, gs, Types.ACTION.UPGRADE_BOMBER);
        } else if (unit.getType() == Types.UNIT.SCOUT) {
            addUpgradeIfFeasible(upgradeActions, unit, gs, Types.ACTION.UPGRADE_BOMBER);
        }
        return upgradeActions;
    }

    private void addUpgradeIfFeasible(LinkedList<Action> upgradeActions, Unit unit, GameState gs, Types.ACTION actionType) {
        Upgrade action = new Upgrade(actionType, unit.getActorId());
        if (action.isFeasible(gs)) {
            upgradeActions.add(action);
        }
    }
}

