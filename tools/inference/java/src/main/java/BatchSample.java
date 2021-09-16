/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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
