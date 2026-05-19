import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

/**
 * Tournament wrapper for fail-fast debugging with automatic log capture.
 */
public class DebugTournament {

    public static void main(String[] args) throws Exception {
        String configPath = args.length > 0 ? args[0] : "tournament.json";
        Path logDir = Paths.get("debug-logs");
        Files.createDirectories(logDir);

        String timestamp = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date());
        String configName = Paths.get(configPath).getFileName().toString().replaceAll("[^A-Za-z0-9._-]", "_");
        Path logPath = logDir.resolve("tournament-debug-" + configName + "-" + timestamp + ".log");

        PrintStream originalOut = System.out;
        PrintStream originalErr = System.err;

        System.setProperty("tribes.debug.fail_fast_move", "true");
        System.setProperty("tribes.debug.fail_fast_build", "true");
        System.setProperty("tribes.debug.stop_on_game_error", "true");

        try (PrintStream logStream = new PrintStream(new FileOutputStream(logPath.toFile()), true, "UTF-8");
             PrintStream teeOut = new PrintStream(new TeeOutputStream(originalOut, logStream), true, "UTF-8");
             PrintStream teeErr = new PrintStream(new TeeOutputStream(originalErr, logStream), true, "UTF-8")) {

            System.setOut(teeOut);
            System.setErr(teeErr);

            System.out.println("DebugTournament starting");
            System.out.println("Config: " + Paths.get(configPath).toAbsolutePath());
            System.out.println("Log: " + logPath.toAbsolutePath());
            System.out.println("Args: " + Arrays.toString(args));
            System.out.println("System properties: tribes.debug.fail_fast_move=true, tribes.debug.fail_fast_build=true, tribes.debug.stop_on_game_error=true");

            try {
                Tournament.main(new String[]{configPath});
                System.out.println("DebugTournament finished without fail-fast errors.");
            } catch (Throwable t) {
                System.err.println("DebugTournament aborted due to an issue. See details above.");
                t.printStackTrace(System.err);
                throw t;
            }
        } finally {
            System.setOut(originalOut);
            System.setErr(originalErr);
            originalOut.println("DebugTournament log written to " + logPath.toAbsolutePath());
        }
    }

    private static final class TeeOutputStream extends OutputStream {
        private final OutputStream first;
        private final OutputStream second;

        private TeeOutputStream(OutputStream first, OutputStream second) {
            this.first = first;
            this.second = second;
        }

        @Override
        public void write(int b) throws IOException {
            first.write(b);
            second.write(b);
        }

        @Override
        public void write(byte[] b) throws IOException {
            first.write(b);
            second.write(b);
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            first.write(b, off, len);
            second.write(b, off, len);
        }

        @Override
        public void flush() throws IOException {
            first.flush();
            second.flush();
        }
    }
}
