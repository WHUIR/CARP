package dataPreprocessing;

import java.io.*;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;


public class IOUTils {
    public static Pattern pattern = Pattern.compile("[0-9]*");

    public static Set<String> readFile2Set(String path) {
        File file = new File(path);
        BufferedReader in = null;
        Set<String> set = new HashSet<>();

        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = null;

            while ((line = in.readLine()) != null) {
                line = line.trim();
                set.add(line);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            closeReader(in);
        }

        return set;
    }

    public static void writeFile(List<String> list, String path) {
        FileWriter fw = null;
        try {
            fw = new FileWriter(path);
            for (int i = 0; i < list.size(); i++) {
                fw.write(list.get(i).trim() + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            closeWriter(fw);
        }
    }

    public static void writeDocSimMap2File(String path, Map<String, Double> map) {
        FileWriter fw = null;
        try {
            fw = new FileWriter(path);
            for (Map.Entry<String, Double> entry: map.entrySet()){
                StringBuilder sb = new StringBuilder();
                sb.append(entry.getKey()).append("\t").append(entry.getValue());
                fw.write(sb.toString() + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            closeWriter(fw);
        }
    }

    public static void writeTopNWord(int thrhld, String path, Map<String, Double> sortedMap) {
        FileWriter fw = null;
        try {
            fw = new FileWriter(path);
            for (Map.Entry<String, Double> entry : sortedMap.entrySet()) {
                fw.write(entry.getKey() + "\n");
                thrhld--;
                if (thrhld == 0) break;
            }
        } catch (IOException e){
            e.printStackTrace();
        }finally {
         closeWriter(fw);
        }
    }

    public static void writeFile2Label(String path, Map<Integer, String> map){
        FileWriter fw = null;
        try {
            fw = new FileWriter(path);
            for (Map.Entry<Integer, String> entry: map.entrySet()){
                StringBuilder sb = new StringBuilder();
                sb.append(entry.getKey()).append("\t").append(entry.getValue());
                fw.write(sb.toString() + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            closeWriter(fw);
        }
    }

    public static void writeStr2IntMap2File(String path, Map<String, Integer> map) {
        FileWriter fw = null;
        try {
            fw = new FileWriter(path);
            for (Map.Entry<String, Integer> entry: map.entrySet()){
                StringBuilder sb = new StringBuilder();
                sb.append(entry.getKey()).append("\t").append(entry.getValue());
                fw.write(sb.toString() + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            closeWriter(fw);
        }
    }

    public static Set<String> getTrainSet(String path, String separator) {
        File file = new File(path);
        BufferedReader in = null;
        Set<String> trains = new HashSet<>();

        try {

            in = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = null;
            while ((line = in.readLine()) != null) {
                line = line.trim();
                String[] str = line.split(separator);
                trains.add(String.format("%s*%s", str[0].trim(), str[1].trim()));
            }
        }catch (Exception e){
            e.printStackTrace();
        }finally {
            closeReader(in);
        }
        return trains;
    }


    public static void closeReader(Reader reader) {
        try {
            if (reader != null)
                reader.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public static void closeWriter (Writer writer){
        try {
            if (writer != null)
                writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

}
