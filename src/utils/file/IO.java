package utils.file;

import org.json.JSONObject;

import java.io.*;
import java.util.ArrayList;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 04/10/13
 * Time: 16:56
 */
public class IO
{
    /**
     * Default constructor
     */
    public IO(){}

    /**
     * Reads a file and returns its content as a String[]
     * @param filename file to read
     * @return file content as String[], one line per element
     */
    public String[] readFile(String filename)
    {
        ArrayList<String> lines = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(new FileReader(filename))) {
            String line = null;
            while ((line = in.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            System.err.println("Error reading file '" + filename + "': " + e.getMessage());
            return null;
        }
        return lines.toArray(new String[0]);
    }


    /**
     * Writes in a file data passed as a String[]
     * @param filename file to write to.
     * @param lines content to write
     * @return true if all worked
     */
    public boolean writeFile(String filename, ArrayList<String> lines, boolean append)
    {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filename), append))) {
            for(String line : lines)
            {
                writer.write(line + "\n");
            }
        } catch (IOException e) {
            System.err.println("Error writing file '" + filename + "': " + e.getMessage());
            return false;
        }
        return true;
    }



    /**
     * Reads 'filename' and returns it as a JSON Object.
     * @param filename path to the file.
     * @return a JSON Object.
     */
    public JSONObject readJSON(String filename) {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();
            while (line != null) {
                sb.append(line);
                line = br.readLine();
            }
            return new JSONObject(sb.toString());
        } catch (Exception e) {
            System.err.println("Error reading JSON file '" + filename + "': " + e.getMessage());
        }
        return new JSONObject();
    }
}
