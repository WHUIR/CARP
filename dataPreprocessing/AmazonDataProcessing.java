package dataPreprocessing;

import org.json.JSONObject;
import util.AuxiliaryUtil;
import util.IOUTil;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AmazonDataProcessing {
    public static final String PATH = "path of the original data file: such as Musical_Instruments_5.json";
    public static final String[] SEPARATOR = new String[]{"\t", "\\*", " "};
    public String path_prefix;
    public static final String[] KEYS = new String[]{"reviewerID", "asin", "reviewText", "overall", "unixReviewTime"};
    public Pattern pattern = Pattern.compile("^\\d*$");
    public static final int WORD_NUM = 20000;//number of retained word

    public AmazonDataProcessing(String path_prefix) {
        this.path_prefix = path_prefix;
    }

    public List<String> extractRawDataAsList() {
        File file = new File(PATH);
        BufferedReader in = null;
        List<String> lines = new ArrayList<>();

        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = null;
            String userId;
            String productId;


            while ((line = in.readLine()) != null) {
                line = line.trim();
                JSONObject jsonObject = new JSONObject(line);
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < KEYS.length; i++) {
                    if (i != KEYS.length - 1)
                        sb.append(jsonObject.get(KEYS[i])).append(SEPARATOR[0]);
                    else
                        sb.append(jsonObject.get(KEYS[i]));
                }
                lines.add(sb.toString());

            }
        }catch (Exception e) {
            e.printStackTrace();
        }finally {
            if (in != null)
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
        }
        return lines;
    }

    public List<String> cleanReviewData(List<String> lines, Set<String> stopWords) {
        Map<String, Integer> word2Times = new HashMap<>();
        int totalDoc = 0;
        List<String> newLines = new ArrayList<>();

        //remove stop word & punctuations
            String review;
            String[] str;
            String[] words;
            Matcher matcher;

            for (String line : lines) {
                int count = 0;
                str = line.trim().split(SEPARATOR[0]);
                review = str[2].trim().toLowerCase().replaceAll("[\\pP+~$`^=|<>～｀＄＾＋＝｜＜＞￥×]", " ");
                words = review.split(SEPARATOR[2]);
                StringBuilder sb = new StringBuilder();
                sb.append(str[0]).append(SEPARATOR[0]).append(str[1]).append(SEPARATOR[0]);
                for (int idx = 0; idx < words.length; idx++) {
                    if (words[idx].equals("") || stopWords.contains(words[idx]) || words[idx].equals(" "))continue;

                    matcher = pattern.matcher(words[idx]);
                    if (matcher.matches())continue;

                    count++;
                    sb.append(words[idx]).append(" ");
                    word2Times = updtMap(words[idx], word2Times);
                }
                if (count == 0)continue;
                sb.replace(sb.length() - 1, sb.length(), "");
                sb.append(SEPARATOR[0]).append(str[3]).append(SEPARATOR[0]).append(str[4]);
                newLines.add(sb.toString());
                totalDoc++;
            }

        lines.clear();
        for (Map.Entry<String, Integer> entry : word2Times.entrySet())
            if (entry.getValue() >= (totalDoc / 2))
                stopWords.add(entry.getKey());

        for (String line : newLines){
            str = line.trim().split(SEPARATOR[0]);
            review = str[2].trim();
            words = review.split(SEPARATOR[2]);
            StringBuilder sb = new StringBuilder();
            sb.append(str[0]).append(SEPARATOR[0]).append(str[1]).append(SEPARATOR[0]);

            for (int idx = 0; idx < words.length; idx++) {
                if (words[idx].equals("") || stopWords.contains(words[idx]) || words[idx].equals(" "))continue;
                sb.append(words[idx]).append(" ");
            }
            sb.replace(sb.length() - 1, sb.length(), "");
            sb.append(SEPARATOR[0]).append(str[3]).append(SEPARATOR[0]).append(str[4]);
            lines.add(sb.toString());
        }

        return lines;
    }

    public Map<String, Integer> updtMap(String key, Map<String, Integer> map) {
        int count = 0;
        if (map.containsKey(key))
            count = map.get(key);
        map.put(key, count + 1);
        return map;
    }

    public Map<Integer, String> updtMap(int userId, String review, Map<Integer, String> map) {
        String content = "";
        if (map.containsKey(userId))
            content = map.get(userId);
        content += " " + review;
        map.put(userId, content);
        return map;
    }

    public Map<String, Double> updtWordDocFreq(Set<String> words, Map<String, Double> word2docFreq){
        double count = 0.0;
        for (String word : words){
            if (word2docFreq.containsKey(word))
                count = word2docFreq.get(word);
            word2docFreq.put(word, count + 1);
        }
        return word2docFreq;
    }

    public double calIdf(double docFreq, double totalDoc) {
        return Math.log(totalDoc / docFreq);
    }

    public Map<String, Double> getVocaList(String suffix) {
        File file = new File(this.path_prefix + suffix);
        BufferedReader in = null;
        Map<String, Double> map = new HashMap<>();
        Map<String, Integer> word2tf = new HashMap<>();
        Map<String, Double> word2docFreq = new HashMap<>();

        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = null;
            String review;
            String[] str;
            String[] words;
            double totalDoc = 0.0;

            while ((line = in.readLine()) != null) {
                str = line.trim().split(SEPARATOR[0]);
                review = str[2].trim();
                words = review.split(SEPARATOR[2]);
                Set<String> wordSet = new HashSet<>();

                for (int i = 0; i < words.length; i++) {
                    if (words[i].equals(""))continue;

                    wordSet.add(words[i]);
                    word2tf = updtMap(words[i], word2tf);
                }
                word2docFreq = updtWordDocFreq(wordSet, word2docFreq);
                totalDoc++;
            }

            double idf;
            double tf;
            for (Map.Entry<String, Integer> entry : word2tf.entrySet()){
                idf = calIdf(word2docFreq.get(entry.getKey()), totalDoc);
                tf = (double) entry.getValue();
                map.put(entry.getKey(), tf * idf);
            }
        }catch (Exception e) {
            e.printStackTrace();
        }finally {
            if (in != null)
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
        }
        return map;
    }

    public List<String> retainFinalWord(String suffix, Set<String> retainWords) {
        File file = new File(path_prefix + suffix);
        BufferedReader in = null;
        List<String> lines = new ArrayList<>();
        Map<String, Integer> word2Times = new HashMap<>();
        int totalDoc = 0;


        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = null;
            String userId;
            String productId;
            String review;
            String rate;
            String time;
            String[] str;
            String[] words;

            while ((line = in.readLine()) != null) {
                int count = 0;
                str = line.trim().split(SEPARATOR[0]);
                review = str[2].trim();
                words = review.split(SEPARATOR[2]);
                StringBuilder sb = new StringBuilder();
                sb.append(str[0]).append(SEPARATOR[0]).append(str[1]).append(SEPARATOR[0]);

                for (int i = 0; i < words.length; i++) {
                    if (!retainWords.contains(words[i])) continue;

                    sb.append(words[i]).append(" ");
                    count++;
                }
                if (count == 0)continue;
                sb.replace(sb.length() - 1, sb.length(), "");
                sb.append(SEPARATOR[0]).append(str[3]).append(SEPARATOR[0]).append(str[4]);
                lines.add(sb.toString());

            }
        }catch (Exception e){
            e.printStackTrace();
        }finally {
            if (in != null)
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
        }
        return lines;
    }

    public Map<String, Integer> getEntity2IdMap(List<String> lines, int index) {
        Map<String, Integer> entity2IdMap = new HashMap<>();
        String entity;
        int count = 0;
        for (String line : lines) {
            String[] str = line.trim().split(SEPARATOR[0]);
            entity = str[index];
            if (entity2IdMap.containsKey(entity))continue;

            entity2IdMap.put(entity, count++);
        }
        return entity2IdMap;
    }

    public Map<Integer, Map<Integer, String>> updtMaps(int userId, int itemId, double value, String time, Map<Integer, Map<Integer, String>> map) {
        Map<Integer, String> item2ValueMap = new HashMap<>();
        if (map.containsKey(userId))
            item2ValueMap = map.get(userId);
        item2ValueMap.put(itemId, value + "*" + time);
        map.put(userId, item2ValueMap);
        return map;
    }

    public Map<Integer, Map<Integer, String>> updtMaps(int userId, int itemId, String value, Map<Integer, Map<Integer, String>> map) {
        Map<Integer, String> item2ValueMap = new HashMap<>();
        if (map.containsKey(userId))
            item2ValueMap = map.get(userId);
        item2ValueMap.put(itemId, value);
        map.put(userId, item2ValueMap);
        return map;
    }

    public Map<Integer, Map<Integer, String>> extractRawInteraction(String inputSuffix, Map<String, Integer> user2IdMap, Map<String, Integer> item2IdMap){
        File file = new File(this.path_prefix + inputSuffix);
        BufferedReader in = null;
        Map<Integer, Map<Integer, String>> user2item2rateMap = new HashMap<>();

        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = null;
            int userId;
            int productId;
            double rate;
            String time;

            while ((line = in.readLine()) != null) {
                String[] str = line.trim().split(SEPARATOR[0]);
                userId = user2IdMap.get(str[0].trim());
                productId = item2IdMap.get(str[1].trim());
                rate = Double.parseDouble(str[3].trim());
                time = str[4].trim();
                user2item2rateMap = updtMaps(userId, productId, rate, time, user2item2rateMap);
            }
        }catch (Exception e) {
            e.printStackTrace();
        }finally {
            if (in != null)
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
        }

        return user2item2rateMap;
    }

    public void writeInteraction(String suffix, Map<Integer, Map<Integer, String>> map) throws IOException {
        FileWriter fw = null;

        try {
            fw = new FileWriter(this.path_prefix + suffix);
            int userId;

            for (Map.Entry<Integer, Map<Integer, String>> entry : map.entrySet()) {
                userId = entry.getKey();

                for (Map.Entry<Integer, String> entry1 : entry.getValue().entrySet()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(userId).append(SEPARATOR[0]).append(entry1.getKey()).append(SEPARATOR[0]);
                    String[] rate2Time = entry1.getValue().trim().split(SEPARATOR[1]);
                    sb.append(Double.parseDouble(rate2Time[0].trim())).append(SEPARATOR[0]).append(rate2Time[1].trim());
                    fw.write(sb.toString() + "\n");
                }
            }
        }catch (Exception e) {
            e.printStackTrace();
        }finally {
            if (fw != null)
                fw.close();
        }
    }

    public Map<Integer, Map<Integer, String>> retrieveUserItemPair(String suffix, Map<String, Integer> user2Id, Map<String, Integer> item2Id) {
        File file = new File(this.path_prefix + suffix);
        BufferedReader in = null;
        Map<Integer, Map<Integer, String>> map = new HashMap<>();

        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = null;
            int itemId;
            int userId;
            double value;
            String time;
            int count = 0;

            while ((line = in.readLine()) != null) {
                String[] str = line.trim().split(SEPARATOR[0]);

                itemId = Integer.parseInt(str[1].trim());
                userId = Integer.parseInt(str[0].trim());
                value = Double.parseDouble(str[2].trim());
                time = str[3].trim();
                map = updtMaps(userId, itemId, value, time, map);
            }
        }catch (Exception e) {
            e.printStackTrace();
        }
        return map;
    }

    public Map<Integer, Map<Integer, String>> retrieveItemUserPair(String suffix, Map<String, Integer> user2Id, Map<String, Integer> item2Id) {
        File file = new File(this.path_prefix + suffix);
        BufferedReader in = null;
        Map<Integer, Map<Integer, String>> map = new HashMap<>();

        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = null;
            int itemId;
            int userId;
            double value;
            String time;
            int count = 0;

            while ((line = in.readLine()) != null) {
                String[] str = line.trim().split(SEPARATOR[0]);

                itemId = Integer.parseInt(str[1].trim());
                userId = Integer.parseInt(str[0].trim());
                value = Double.parseDouble(str[2].trim());
                time = str[3].trim();
                map = updtMaps(itemId, userId, value, time, map);
            }
        }catch (Exception e) {
            e.printStackTrace();
        }
        return map;
    }

    public Set<String> getInteraction(Map<Integer, Map<Integer, String>> user2item2rateMap) {
        Set<String> set = new HashSet<>();
        for (Map.Entry<Integer, Map<Integer, String>> entry : user2item2rateMap.entrySet())
            for (Map.Entry<Integer, String> entry1 : entry.getValue().entrySet())
                set.add(entry.getKey() + "*" + entry1.getKey());
        return set;
    }

    public void getTrainTestSet(double thrhld, int userNum, int itemNum, String trainSuffix, String testSuffix, Map<Integer, Map<Integer, String>> user2item2rateMap, Map<Integer, Map<Integer, String>> item2user2rateMap) throws IOException {
        Map<Integer, Map<Integer, String>> trainMap = new HashMap<>();
        Map<Integer, Map<Integer, String>> testMap = new HashMap<>();
        Map<Integer, String> subMap;
        Set<String> allInteraction = getInteraction(user2item2rateMap);
        Set<String> alrdyTrainSet = new HashSet<>();
        Set<Integer> alrdItem = new HashSet<>();
        double totalCases = allInteraction.size();
        double trainNum = Math.ceil(totalCases * thrhld);
        double testNum = totalCases - trainNum;

        for (int i = 0; i < userNum; i++) {
            subMap = user2item2rateMap.get(i);
            for (Map.Entry<Integer, String> entry : subMap.entrySet()) {
                trainMap = updtMaps(i, entry.getKey(), entry.getValue(), trainMap);
                alrdyTrainSet.add(i + "*" + entry.getKey());
                alrdItem.add(entry.getKey());
                trainNum--;
                break;
            }
        }

        for (int i = 0; i < itemNum; i++) {
            if (alrdItem.contains(i)) continue;
            subMap = item2user2rateMap.get(i);
            for (Map.Entry<Integer, String> entry : subMap.entrySet()) {
                trainMap = updtMaps(entry.getKey(), i, entry.getValue(), trainMap);
                alrdyTrainSet.add(entry.getKey() + "*" + i);
                alrdItem.add(i);
                trainNum--;
                break;
            }
        }

        thrhld = trainNum / (trainNum + testNum);
        Random rand = new Random();
        double seed;

        for (String key : allInteraction){
            if (alrdyTrainSet.contains(key))continue;

            seed = rand.nextDouble();
            String[] str = key.split(SEPARATOR[1]);
            if (seed <= thrhld){
                trainMap = updtMaps(Integer.parseInt(str[0]), Integer.parseInt(str[1]), user2item2rateMap.get(Integer.parseInt(str[0])).get(Integer.parseInt(str[1])), trainMap);
                alrdyTrainSet.add(key);
            }else {
                testMap = updtMaps(Integer.parseInt(str[0]), Integer.parseInt(str[1]), user2item2rateMap.get(Integer.parseInt(str[0])).get(Integer.parseInt(str[1])), testMap);
            }

        }

        writeInteraction(trainSuffix, trainMap);
        writeInteraction(testSuffix, testMap);
    }

    public Map<Integer, String> retrieveUserContent(int idx, String suffix, Set<String> tests, Map<String, Integer> entity2Id, Map<String, Integer> user2Id, Map<String, Integer> item2Id){
        File file = new File(this.path_prefix + suffix);
        BufferedReader in = null;
        Map<Integer, String> map = new HashMap<>();

        try {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = null;
            int entityId;
            int userId;
            int itemId;
            String review;
            int count = 0;
            String key;
            Set<String> set = new HashSet<>();

            while ((line = in.readLine()) != null) {
                line = line.trim();
                String[] str = line.split(SEPARATOR[0]);
                entityId = entity2Id.get(str[idx].trim());
                userId = user2Id.get(str[0].trim());
                itemId = item2Id.get(str[1].trim());
                key = String.format("%d%s%d", userId, "*", itemId);
                if (set.contains(key) || tests.contains(key)) continue;

                set.add(key);
                review = str[2].trim();
                map = updtMap(entityId, review, map);
                count += 1;
                if (count % 100000 == 0)
                    System.out.println("100000 finish");
            }

        }catch (Exception e) {
            e.printStackTrace();
        }finally {
            if (in != null)
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
        }
        return map;
    }

    public Map<String, Integer> wordStatics(String seperator, Map<Integer, String> item2review) {
        String content;
        Map<String, Integer> word2IdMap = new HashMap<>();
        int count = 0;
        for (Map.Entry<Integer, String> entry : item2review.entrySet()){
            content = entry.getValue();
            String[] words = content.split(SEPARATOR[2]);
            for (int i = 0; i < words.length; i++) {
                if (word2IdMap.containsKey(words[i].trim()) || words[i].trim().equals(""))continue;

                word2IdMap.put(words[i].trim(), count++);
            }
        }
        return word2IdMap;
    }

    public static void main(String[] args) throws IOException {
        AmazonDataProcessing amazonFileExtraction = new AmazonDataProcessing("output path");
        List<String> lines = amazonFileExtraction.extractRawDataAsList();
        Set<String> stopWords = IOUTils.readFile2Set("stopword path");
        lines = amazonFileExtraction.cleanReviewData(lines, stopWords);
        IOUTils.writeFile(lines, amazonFileExtraction.path_prefix + "CleanVersion.out");

        Map<String, Double> map = amazonFileExtraction.getVocaList("CleanVersion.out");
        Map<String, Double> sortedMap = AuxiliaryUtil.sortByValueDesc(map);
        IOUTils.writeDocSimMap2File(amazonFileExtraction.path_prefix + "word2tf-idf.out", map);
        IOUTils.writeTopNWord(WORD_NUM, amazonFileExtraction.path_prefix + "retainWordVoca.out", sortedMap);
        Set<String> retainWords = IOUTil.readFile2Set(amazonFileExtraction.path_prefix + "retainWordVoca.out");
        lines = amazonFileExtraction.retainFinalWord("CleanVersion.out", retainWords);
        IOUTil.writeFile(lines, amazonFileExtraction.path_prefix+"FinalCleanVersion.out");
//        List<String> lines = IOUTil.readFile2List(amazonFileExtraction.path_prefix + "FinalCleanVersion.out");
        //mapping user & item 2 id
        Map<String, Integer> user2IdMap = amazonFileExtraction.getEntity2IdMap(lines, 0);
        Map<String, Integer> item2IdMap = amazonFileExtraction.getEntity2IdMap(lines, 1);
        IOUTil.writeStr2IntMap2File(amazonFileExtraction.path_prefix + "user2IdMapping.out", user2IdMap);
        IOUTil.writeStr2IntMap2File(amazonFileExtraction.path_prefix + "item2IdMapping.out", item2IdMap);

        //split the dataset
        Map<Integer, Map<Integer, String>> user2item2rateMap = amazonFileExtraction.extractRawInteraction("FinalCleanVersion.out", user2IdMap, item2IdMap);
        amazonFileExtraction.writeInteraction("TotalInteraction.out", user2item2rateMap);
        user2item2rateMap = amazonFileExtraction.retrieveUserItemPair("TotalInteraction.out", user2IdMap, item2IdMap);
        Map<Integer, Map<Integer, String>> item2user2rateMap = amazonFileExtraction.retrieveItemUserPair("TotalInteraction.out", user2IdMap, item2IdMap);
        amazonFileExtraction.getTrainTestSet(0.8, user2IdMap.size(), item2IdMap.size(), "TrainInteraction.out", "TestInteraction.out", user2item2rateMap, item2user2rateMap);

        //get validation set
        user2item2rateMap = amazonFileExtraction.retrieveUserItemPair("TrainInteraction.out", user2IdMap, item2IdMap);
        item2user2rateMap = amazonFileExtraction.retrieveItemUserPair("TrainInteraction.out", user2IdMap, item2IdMap);
        amazonFileExtraction.getTrainTestSet(0.9, user2IdMap.size(), item2IdMap.size(), "ValTrainInteraction.out", "ValInteraction.out", user2item2rateMap, item2user2rateMap);

        //get user/item review documents
        Set<String> tests = IOUTils.getTrainSet(amazonFileExtraction.path_prefix + "TestInteraction.out", SEPARATOR[0]);
        Map<Integer, String> user2Reviews = amazonFileExtraction.retrieveUserContent(0, "FinalCleanVersion.out", tests, user2IdMap, user2IdMap, item2IdMap);
        IOUTil.writeFile2Label(amazonFileExtraction.path_prefix + "UserReviews.out", user2Reviews);

        Map<Integer, String> item2Reviews = amazonFileExtraction.retrieveUserContent(1, "FinalCleanVersion.out", tests, item2IdMap, user2IdMap, item2IdMap);
        IOUTil.writeFile2Label(amazonFileExtraction.path_prefix + "ItemReviews.out", item2Reviews);

        //get word2id document
        Map<String, Integer> word2IdMap = amazonFileExtraction.wordStatics(SEPARATOR[0], user2Reviews);
        System.out.println(word2IdMap.size());
        IOUTil.writeStr2IntMap2File(amazonFileExtraction.path_prefix + "WordDict.out", word2IdMap);
    }
}
