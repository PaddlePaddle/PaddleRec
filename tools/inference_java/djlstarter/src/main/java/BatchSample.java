/**
 * descript: BatchSample
 * author: wangbin44@baidu.com
 * date: 2021.8.2
 */

import java.util.*;

public final class BatchSample {
    public void clear() {
        features.clear();
        featureCnts.clear();
    }

    public int length(int slotId) {
        int len = 0;
        for (int sampleIdx = 0; sampleIdx < featureCnts.get(slotId).size(); sampleIdx++) {
            len += featureCnts.get(slotId).get(sampleIdx);
        }
        return len;
    }

    public int size() {
        if (features.size() == featureCnts.size()) {
            return features.size();
        } else {
            return 0;
        }
    }

    public HashMap<Integer, ArrayList<ArrayList<Integer>>> features = new HashMap<Integer, ArrayList<ArrayList<Integer>>>(); // key: slotId
    public HashMap<Integer, ArrayList<Integer>> featureCnts = new HashMap<Integer, ArrayList<Integer>>();
};
