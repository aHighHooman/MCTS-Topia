package core.levelgen;

import core.Types;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Headless sampler for generated map statistics.
 *
 * Defaults to 1,000 maps for each size/type pair: 6 sizes * 6 map types * 1,000 = 36,000 maps.
 * Use --maps-per-config 10000 for 360,000 maps.
 */
public final class MapGenerationStats {

    private static final Types.TRIBE[] DEFAULT_TRIBES = new Types.TRIBE[]{
            Types.TRIBE.IMPERIUS,
            Types.TRIBE.BARDUR,
            Types.TRIBE.KICKOO,
            Types.TRIBE.XIN_XI
    };

    private MapGenerationStats() {
    }

    public static void main(String[] args) throws IOException {
        Config config = Config.parse(args);
        File outDir = new File(config.outDir);
        if (!outDir.exists() && !outDir.mkdirs()) {
            throw new IOException("Failed to create output directory: " + outDir.getAbsolutePath());
        }

        File detailFile = new File(outDir, "map_generation_detail.csv");
        long started = System.currentTimeMillis();
        int total = Types.MAP_SIZE.values().length * Types.MAP_TYPE.values().length * config.mapsPerConfig;
        int written = 0;

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(detailFile))) {
            writeHeader(writer);
            ExecutorService executor = Executors.newFixedThreadPool(config.threads);
            CompletionService<String> completion = new ExecutorCompletionService<>(executor);
            for (Types.MAP_SIZE size : Types.MAP_SIZE.values()) {
                for (Types.MAP_TYPE mapType : Types.MAP_TYPE.values()) {
                    for (int sample = 0; sample < config.mapsPerConfig; sample++) {
                        final Types.MAP_SIZE taskSize = size;
                        final Types.MAP_TYPE taskMapType = mapType;
                        final int taskSample = sample;
                        completion.submit(() -> generateRow(config, taskSize, taskMapType, taskSample));
                    }
                }
            }

            try {
                while (written < total) {
                    Future<String> future = completion.take();
                    writer.write(future.get());
                    writer.newLine();
                    written++;
                    if (written % Math.max(1, total / 36) == 0 || written == total) {
                        System.out.println("Wrote " + written + "/" + total + " rows");
                    }
                }
            } catch (Exception e) {
                executor.shutdownNow();
                throw new IllegalStateException("Map statistics generation failed.", e);
            } finally {
                executor.shutdown();
            }
        }

        long elapsedMs = System.currentTimeMillis() - started;
        System.out.println("Wrote " + written + " map rows to " + detailFile.getAbsolutePath()
                + " in " + elapsedMs + " ms.");
    }

    private static String generateRow(Config config, Types.MAP_SIZE size, Types.MAP_TYPE mapType, int sample) {
        long seed = config.seed + seedOffset(size, mapType, sample);
        LevelGenerator generator = new LevelGenerator(seed);
        generator.init(size.getSideLength(), mapType, config.tribes);
        generator.generate();
        return Stats.from(generator.gelLevelLines(), size, mapType, seed, sample, config.tribes).toCsvLine();
    }

    private static long seedOffset(Types.MAP_SIZE size, Types.MAP_TYPE mapType, int sample) {
        return ((long) size.ordinal() * 10_000_000L) + ((long) mapType.ordinal() * 1_000_000L) + sample;
    }

    private static void writeHeader(BufferedWriter writer) throws IOException {
        ArrayList<String> columns = new ArrayList<>();
        columns.add("seed");
        columns.add("sample");
        columns.add("map_size");
        columns.add("side_length");
        columns.add("tile_count");
        columns.add("map_type");
        columns.add("tribe_roster");
        for (Types.TERRAIN terrain : Types.TERRAIN.values()) {
            columns.add("terrain_" + terrain.name().toLowerCase(Locale.ROOT));
        }
        for (Types.RESOURCE resource : Types.RESOURCE.values()) {
            columns.add("resource_" + resource.name().toLowerCase(Locale.ROOT));
        }
        columns.add("settlements");
        columns.add("capitals");
        columns.add("villages");
        columns.add("water_tiles");
        columns.add("land_tiles");
        columns.add("water_ratio");
        columns.add("land_ratio");
        columns.add("coastal_settlements");
        columns.add("coastal_settlement_ratio");
        columns.add("land_components");
        columns.add("largest_land_component");
        columns.add("largest_land_component_share");
        columns.add("edge_land_tiles");
        columns.add("edge_land_ratio");
        columns.add("mean_capital_distance");
        columns.add("min_capital_distance");
        columns.add("ruins_per_100_tiles");
        columns.add("resources_per_100_tiles");
        writer.write(String.join(",", columns));
        writer.newLine();
    }

    private static final class Config {
        private int mapsPerConfig = 1000;
        private long seed = 1000003L;
        private String outDir = "tmp/mapgen-stats";
        private int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        private Types.TRIBE[] tribes = DEFAULT_TRIBES;

        private static Config parse(String[] args) {
            Config config = new Config();
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if ("--maps-per-config".equals(arg)) {
                    config.mapsPerConfig = Integer.parseInt(requireValue(args, ++i, arg));
                } else if ("--seed".equals(arg)) {
                    config.seed = Long.parseLong(requireValue(args, ++i, arg));
                } else if ("--out-dir".equals(arg)) {
                    config.outDir = requireValue(args, ++i, arg);
                } else if ("--threads".equals(arg)) {
                    config.threads = Integer.parseInt(requireValue(args, ++i, arg));
                } else if ("--tribes".equals(arg)) {
                    config.tribes = parseTribes(requireValue(args, ++i, arg));
                } else if ("--help".equals(arg) || "-h".equals(arg)) {
                    printHelpAndExit();
                } else {
                    throw new IllegalArgumentException("Unknown argument: " + arg);
                }
            }
            if (config.mapsPerConfig <= 0) {
                throw new IllegalArgumentException("--maps-per-config must be positive.");
            }
            if (config.tribes.length == 0) {
                throw new IllegalArgumentException("--tribes must include at least one tribe.");
            }
            if (config.threads <= 0) {
                throw new IllegalArgumentException("--threads must be positive.");
            }
            return config;
        }

        private static String requireValue(String[] args, int index, String flag) {
            if (index >= args.length) {
                throw new IllegalArgumentException("Missing value after " + flag);
            }
            return args[index];
        }

        private static Types.TRIBE[] parseTribes(String value) {
            String[] pieces = value.split(",");
            Types.TRIBE[] tribes = new Types.TRIBE[pieces.length];
            for (int i = 0; i < pieces.length; i++) {
                String normalized = pieces[i].trim().toUpperCase(Locale.ROOT).replace(' ', '_').replace('-', '_');
                tribes[i] = Types.TRIBE.valueOf(normalized);
            }
            return tribes;
        }

        private static void printHelpAndExit() {
            System.out.println("Usage: java -cp \"out;lib/json.jar\" core.levelgen.MapGenerationStats "
                    + "[--maps-per-config 1000] [--seed 1000003] [--out-dir tmp/mapgen-stats] "
                    + "[--threads 7] [--tribes IMPERIUS,BARDUR,KICKOO,XIN_XI]");
            System.exit(0);
        }
    }

    private static final class Stats {
        private final long seed;
        private final int sample;
        private final Types.MAP_SIZE mapSize;
        private final Types.MAP_TYPE mapType;
        private final String tribeRoster;
        private final int side;
        private final int tileCount;
        private final int[] terrainCounts;
        private final int[] resourceCounts;
        private final ArrayList<Integer> capitals;
        private final ArrayList<Integer> settlements;
        private final int coastalSettlements;
        private final int landComponents;
        private final int largestLandComponent;
        private final int edgeLandTiles;

        private Stats(long seed, int sample, Types.MAP_SIZE mapSize, Types.MAP_TYPE mapType, String tribeRoster,
                      int side, int tileCount, int[] terrainCounts, int[] resourceCounts,
                      ArrayList<Integer> capitals, ArrayList<Integer> settlements, int coastalSettlements,
                      int landComponents, int largestLandComponent, int edgeLandTiles) {
            this.seed = seed;
            this.sample = sample;
            this.mapSize = mapSize;
            this.mapType = mapType;
            this.tribeRoster = tribeRoster;
            this.side = side;
            this.tileCount = tileCount;
            this.terrainCounts = terrainCounts;
            this.resourceCounts = resourceCounts;
            this.capitals = capitals;
            this.settlements = settlements;
            this.coastalSettlements = coastalSettlements;
            this.landComponents = landComponents;
            this.largestLandComponent = largestLandComponent;
            this.edgeLandTiles = edgeLandTiles;
        }

        private static Stats from(String[] lines, Types.MAP_SIZE mapSize, Types.MAP_TYPE mapType,
                                  long seed, int sample, Types.TRIBE[] tribes) {
            int side = mapSize.getSideLength();
            int tileCount = side * side;
            Types.TERRAIN[] terrain = new Types.TERRAIN[tileCount];
            Types.RESOURCE[] resource = new Types.RESOURCE[tileCount];
            int[] terrainCounts = new int[Types.TERRAIN.values().length];
            int[] resourceCounts = new int[Types.RESOURCE.values().length];
            ArrayList<Integer> capitals = new ArrayList<>();
            ArrayList<Integer> settlements = new ArrayList<>();

            for (int row = 0; row < side; row++) {
                String[] cells = lines[row].split(",");
                for (int col = 0; col < side; col++) {
                    int index = row * side + col;
                    String[] parts = cells[col].split(":");
                    Types.TERRAIN terrainType = Types.TERRAIN.getType(parts[0].charAt(0));
                    Types.RESOURCE resourceType = null;
                    if (parts.length > 1 && parts[1].length() > 0 && parts[1].charAt(0) != ' ') {
                        resourceType = Types.RESOURCE.getType(parts[1].charAt(0));
                    }
                    terrain[index] = terrainType;
                    resource[index] = resourceType;
                    if (terrainType != null) {
                        terrainCounts[terrainType.ordinal()]++;
                    }
                    if (resourceType != null) {
                        resourceCounts[resourceType.ordinal()]++;
                    }
                    if (terrainType == Types.TERRAIN.CITY) {
                        capitals.add(index);
                        settlements.add(index);
                    } else if (terrainType == Types.TERRAIN.VILLAGE) {
                        settlements.add(index);
                    }
                }
            }

            int coastalSettlements = 0;
            for (Integer settlement : settlements) {
                if (isCoastal(settlement, side, terrain)) {
                    coastalSettlements++;
                }
            }

            ComponentStats componentStats = componentStats(side, terrain);
            int edgeLandTiles = countEdgeLand(side, terrain);
            return new Stats(seed, sample, mapSize, mapType, roster(tribes), side, tileCount, terrainCounts,
                    resourceCounts, capitals, settlements, coastalSettlements, componentStats.count,
                    componentStats.largest, edgeLandTiles);
        }

        private String toCsvLine() {
            ArrayList<String> values = new ArrayList<>();
            int waterTiles = terrainCounts[Types.TERRAIN.SHALLOW_WATER.ordinal()]
                    + terrainCounts[Types.TERRAIN.DEEP_WATER.ordinal()];
            int landTiles = tileCount - waterTiles;
            int villages = terrainCounts[Types.TERRAIN.VILLAGE.ordinal()];
            int resources = 0;
            for (int count : resourceCounts) {
                resources += count;
            }
            values.add(Long.toString(seed));
            values.add(Integer.toString(sample));
            values.add(mapSize.name());
            values.add(Integer.toString(side));
            values.add(Integer.toString(tileCount));
            values.add(mapType.name());
            values.add(tribeRoster);
            for (int count : terrainCounts) {
                values.add(Integer.toString(count));
            }
            for (int count : resourceCounts) {
                values.add(Integer.toString(count));
            }
            values.add(Integer.toString(settlements.size()));
            values.add(Integer.toString(capitals.size()));
            values.add(Integer.toString(villages));
            values.add(Integer.toString(waterTiles));
            values.add(Integer.toString(landTiles));
            values.add(format(waterTiles / (double) tileCount));
            values.add(format(landTiles / (double) tileCount));
            values.add(Integer.toString(coastalSettlements));
            values.add(format(settlements.isEmpty() ? 0.0 : coastalSettlements / (double) settlements.size()));
            values.add(Integer.toString(landComponents));
            values.add(Integer.toString(largestLandComponent));
            values.add(format(landTiles == 0 ? 0.0 : largestLandComponent / (double) landTiles));
            values.add(Integer.toString(edgeLandTiles));
            values.add(format(edgeTileCount(side) == 0 ? 0.0 : edgeLandTiles / (double) edgeTileCount(side)));
            values.add(format(meanCapitalDistance()));
            values.add(format(minCapitalDistance()));
            values.add(format(resourceCounts[Types.RESOURCE.RUINS.ordinal()] * 100.0 / tileCount));
            values.add(format(resources * 100.0 / tileCount));
            return String.join(",", values);
        }

        private double meanCapitalDistance() {
            if (capitals.size() < 2) {
                return 0.0;
            }
            int pairs = 0;
            int total = 0;
            for (int i = 0; i < capitals.size(); i++) {
                for (int j = i + 1; j < capitals.size(); j++) {
                    total += distance(capitals.get(i), capitals.get(j), side);
                    pairs++;
                }
            }
            return total / (double) pairs;
        }

        private double minCapitalDistance() {
            if (capitals.size() < 2) {
                return 0.0;
            }
            int min = Integer.MAX_VALUE;
            for (int i = 0; i < capitals.size(); i++) {
                for (int j = i + 1; j < capitals.size(); j++) {
                    min = Math.min(min, distance(capitals.get(i), capitals.get(j), side));
                }
            }
            return min;
        }
    }

    private static boolean isCoastal(int index, int side, Types.TERRAIN[] terrain) {
        for (Integer neighbour : crossNeighbors(index, side)) {
            if (isWater(terrain[neighbour])) {
                return true;
            }
        }
        return false;
    }

    private static ComponentStats componentStats(int side, Types.TERRAIN[] terrain) {
        boolean[] visited = new boolean[terrain.length];
        int components = 0;
        int largest = 0;
        for (int index = 0; index < terrain.length; index++) {
            if (visited[index] || isWater(terrain[index])) {
                continue;
            }
            components++;
            int size = 0;
            ArrayDeque<Integer> queue = new ArrayDeque<>();
            queue.add(index);
            visited[index] = true;
            while (!queue.isEmpty()) {
                int current = queue.removeFirst();
                size++;
                for (Integer neighbour : crossNeighbors(current, side)) {
                    if (!visited[neighbour] && !isWater(terrain[neighbour])) {
                        visited[neighbour] = true;
                        queue.add(neighbour);
                    }
                }
            }
            largest = Math.max(largest, size);
        }
        return new ComponentStats(components, largest);
    }

    private static int countEdgeLand(int side, Types.TERRAIN[] terrain) {
        HashSet<Integer> edge = new HashSet<>();
        for (int i = 0; i < side; i++) {
            edge.add(i);
            edge.add((side - 1) * side + i);
            edge.add(i * side);
            edge.add(i * side + side - 1);
        }
        int count = 0;
        for (Integer index : edge) {
            if (!isWater(terrain[index])) {
                count++;
            }
        }
        return count;
    }

    private static int edgeTileCount(int side) {
        return side <= 1 ? side : (side * 4) - 4;
    }

    private static boolean isWater(Types.TERRAIN terrain) {
        return terrain == Types.TERRAIN.SHALLOW_WATER || terrain == Types.TERRAIN.DEEP_WATER;
    }

    private static ArrayList<Integer> crossNeighbors(int index, int side) {
        ArrayList<Integer> neighbours = new ArrayList<>();
        int row = index / side;
        int col = index % side;
        if (row > 0) {
            neighbours.add(index - side);
        }
        if (row < side - 1) {
            neighbours.add(index + side);
        }
        if (col > 0) {
            neighbours.add(index - 1);
        }
        if (col < side - 1) {
            neighbours.add(index + 1);
        }
        return neighbours;
    }

    private static int distance(int a, int b, int side) {
        int ax = a % side;
        int ay = a / side;
        int bx = b % side;
        int by = b / side;
        return Math.max(Math.abs(ax - bx), Math.abs(ay - by));
    }

    private static String roster(Types.TRIBE[] tribes) {
        String[] names = new String[tribes.length];
        for (int i = 0; i < tribes.length; i++) {
            names[i] = tribes[i].name();
        }
        return String.join("|", Arrays.asList(names));
    }

    private static String format(double value) {
        return String.format(Locale.ROOT, "%.6f", value);
    }

    private static final class ComponentStats {
        private final int count;
        private final int largest;

        private ComponentStats(int count, int largest) {
            this.count = count;
            this.largest = largest;
        }
    }
}
