package core;

import core.actors.Tribe;

/**
 * Shared unlock and research rules used by action feasibility checks and factories.
 * Keeping these helpers centralized reduces drift between legal-action generation and
 * the individual action implementations that bots rely on.
 */
public final class UnlockRules {

    private UnlockRules() {}

    public static boolean hasTechnology(Tribe tribe, Types.TECHNOLOGY requirement) {
        return requirement == null || tribe.getTechTree().isResearched(requirement);
    }

    public static boolean isActionUnlocked(Tribe tribe, Types.ACTION action) {
        return hasTechnology(tribe, action.getTechnologyRequirement());
    }

    public static boolean isUnitUnlocked(Tribe tribe, Types.UNIT unitType) {
        return hasTechnology(tribe, unitType.getTechnologyRequirement());
    }

    public static boolean isBuildingUnlocked(Tribe tribe, Types.BUILDING buildingType) {
        return hasTechnology(tribe, buildingType.getTechnologyRequirement());
    }

    public static boolean canResearch(Tribe tribe, Types.TECHNOLOGY technology) {
        return tribe.getStars() >= technology.getCost(tribe.getNumCities(), tribe.getTechTree()) &&
                tribe.getTechTree().isResearchable(technology);
    }
}
