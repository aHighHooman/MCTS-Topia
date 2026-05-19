package core;

import java.io.File;

final class PolytopiaAssets {
    private static final String ROOT = "img/polytopia-public/";

    private static final String[] UNIT_TRIBE = {
            "Xin-xi/default/Xin-xi_default_",
            "Imperius/default/Imperius_default_",
            "Bardur/default/Bardur_default_",
            "Oumaji/default/Oumaji_default_",
            "Kickoo/default/Kickoo_default_",
            "Hoodrick/default/Hoodrick_default_",
            "Luxidoor/default/Luxidoor_default_",
            "Vengir/default/Vengir_default_",
            "Zebasi/default/Zebasi_default_",
            "Ai-Mo/default/Ai-Mo_default_",
            "Quetzali/default/Quetzali_default_",
            "Y\u0103dakk/default/Y\u0103dakk_default_"
    };

    private PolytopiaAssets() {
    }

    static String terrain(Types.TERRAIN terrain, String suffix) {
        switch (terrain) {
            case PLAIN:
                return ROOT + "derived/ground_2_top.png";
            case FOREST:
                return ROOT + "Terrain/Forests/Forest_2.png";
            case MOUNTAIN:
                return ROOT + "Terrain/Mountains/mountain_2.png";
            case VILLAGE:
                return ROOT + "Buildings/common/Tribe.png";
            case CITY:
                return ROOT + "Buildings/Imperius/Default/Houses/House_2_1.png";
            case SHALLOW_WATER:
                return ROOT + "derived/water_top.png";
            case DEEP_WATER:
                return ROOT + "derived/ocean_top.png";
            default:
                return null;
        }
    }

    static String resource(Types.RESOURCE resource, Types.TERRAIN terrain) {
        switch (resource) {
            case FISH:
                return ROOT + "Terrain/Misc/ResourceGFX_fish.png";
            case FRUIT:
                return ROOT + "Fruits/ResourceGFX_fruit_2.png";
            case ANIMAL:
                return ROOT + "Animals/horse0002.png";
            case STARFISH:
                return ROOT + "Terrain/Misc/ResourceGFX_starfish.png";
            case ORE:
                return ROOT + "Terrain/Misc/ResourceGFX_metal.png";
            case CROPS:
                return ROOT + "Terrain/Misc/ResourceGFX_crop.png";
            case RUINS:
                return ROOT + "Terrain/Misc/ResourceGFX_ruin.png";
            case LIGHTHOUSE:
                return ROOT + "Buildings/Imperius/Default/Houses/House_2_7.png";
            default:
                return null;
        }
    }

    static String building(Types.BUILDING building) {
        switch (building) {
            case PORT:
                return ROOT + "Buildings/common/Port.png";
            case MINE:
                return ROOT + "Buildings/common/Mine.png";
            case FORGE:
                return ROOT + "Buildings/common/Forge_4.png";
            case FARM:
                return ROOT + "Buildings/common/Farm.png";
            case WINDMILL:
                return ROOT + "Buildings/common/Windmill_6.png";
            case MARKET:
                return ROOT + "Buildings/common/Market01.png";
            case LUMBER_HUT:
                return ROOT + "Buildings/common/Lumber Hut.png";
            case SAWMILL:
                return ROOT + "Buildings/common/Sawmill_8.png";
            case TEMPLE:
                return ROOT + "Buildings/common/Temple_5.png";
            case WATER_TEMPLE:
                return ROOT + "Buildings/common/Water Temple_5.png";
            case FOREST_TEMPLE:
                return ROOT + "Buildings/common/Forest Temple_5.png";
            case MOUNTAIN_TEMPLE:
                return ROOT + "Buildings/common/Mountain Temple_5.png";
            case ALTAR_OF_PEACE:
                return ROOT + "Buildings/Imperius/Default/Monuments/Monument1_2.png";
            case EMPERORS_TOMB:
                return ROOT + "Buildings/Imperius/Default/Monuments/Monument2_2.png";
            case EYE_OF_GOD:
                return ROOT + "Buildings/Imperius/Default/Monuments/Monument3_2.png";
            case GATE_OF_POWER:
                return ROOT + "Buildings/Imperius/Default/Monuments/Monument4_2.png";
            case GRAND_BAZAR:
                return ROOT + "Buildings/Imperius/Default/Monuments/Monument5_2.png";
            case PARK_OF_FORTUNE:
                return ROOT + "Buildings/Imperius/Default/Monuments/Monument6_2.png";
            case TOWER_OF_WISDOM:
                return ROOT + "Buildings/Imperius/Default/Monuments/Monument7_2.png";
            case EMBASSY:
                return ROOT + "Buildings/common/Tribe.png";
            default:
                return null;
        }
    }

    static String unit(Types.UNIT unit, int tribeKey) {
        String unitName = unitName(unit);
        if (unitName == null || tribeKey < 0 || tribeKey >= UNIT_TRIBE.length) {
            return null;
        }
        String path = ROOT + "Units/" + UNIT_TRIBE[tribeKey] + unitName + ".png";
        return exists(path) ? path : null;
    }

    static String weapon(Types.UNIT unit) {
        switch (unit) {
            case ARCHER:
            case RAFT:
            case SCOUT:
                return ROOT + "Misc/arrow.png";
            case CATAPULT:
            case BOMBER:
                return ROOT + "Misc/rock.png";
            case MIND_BENDER:
                return ROOT + "Misc/magic.png";
            case WARRIOR:
            case RIDER:
            case DEFENDER:
                return ROOT + "Misc/club.png";
            default:
                return ROOT + "Misc/sword.png";
        }
    }

    static String action(Types.ACTION action) {
        switch (action) {
            case ATTACK:
            case INFILTRATE:
                return ROOT + "Misc/attackTarget.png";
            case CAPTURE:
                return ROOT + "Misc/Capture.png";
            case EXAMINE:
                return ROOT + "Misc/Examine.png";
            case HEAL_OTHERS:
                return ROOT + "Misc/heal.png";
            case MOVE:
                return ROOT + "Misc/moveTarget.png";
            case CONVERT:
                return ROOT + "Misc/magic.png";
            default:
                return null;
        }
    }

    private static String water(String base, String suffix) {
        if (suffix == null || suffix.isEmpty()) {
            return ROOT + "Terrain/Water/" + base + ".png";
        }
        boolean left = suffix.contains("left");
        boolean right = suffix.contains("right");
        if (left && right) {
            return ROOT + "Terrain/Water/" + base + "_wall_left_wall_right.png";
        }
        if (left) {
            return ROOT + "Terrain/Water/" + base + "_wall_left.png";
        }
        if (right) {
            return ROOT + "Terrain/Water/" + base + "_wall_right.png";
        }
        return ROOT + "Terrain/Water/" + base + ".png";
    }

    private static String unitName(Types.UNIT unit) {
        switch (unit) {
            case WARRIOR:
                return "Warrior";
            case RIDER:
                return "Rider";
            case DEFENDER:
                return "Defender";
            case SWORDMAN:
                return "Swordsman";
            case ARCHER:
                return "Archer";
            case CATAPULT:
                return "Catapult";
            case KNIGHT:
                return "Knight";
            case MIND_BENDER:
                return "MindBender";
            case RAFT:
            case DINGHY:
                return "Boat";
            case SCOUT:
                return "Scoutship";
            case BOMBER:
                return "Bombership";
            case SUPERUNIT:
                return "Giant";
            case CLOAK:
                return "Cloak";
            case DAGGER:
                return "Dagger";
            case RAMMER:
                return "Rammership";
            case JUGGERNAUT:
                return "Juggernaut";
            case PIRATE:
                return "Pirate";
            default:
                return null;
        }
    }

    private static boolean exists(String path) {
        return new File(path).exists();
    }
}
