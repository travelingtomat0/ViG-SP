#include <unordered_map>
#include <vector>

extern "C" void store_labels(long *labels, int w, int h, char *filename) {
  FILE *f = fopen(filename, "wb");

  int width = w, height = h;
  fwrite(&width, sizeof(width), 1, f);
  fwrite(&height, sizeof(height), 1, f);

  int num = w * h;
  for (int i = 0; i < num;) {
    int curr = labels[i];
    int reps = 0;
    for (; i < num && labels[i] == curr; i++) {
      reps++;
    }
    fwrite(&curr, sizeof(curr), 1, f);
    fwrite(&reps, sizeof(reps), 1, f);
  }

  fflush(f);
  fclose(f);
}

extern "C" void load_labels(char *filename, long *labels) {
  FILE *f = fopen(filename, "rb");

  int width, height;
  fread(&width, sizeof(width), 1, f);
  fread(&height, sizeof(height), 1, f);

  int num = width * height;

  for (int i = 0; i < num;) {
    int curr, reps;
    fread(&curr, sizeof(curr), 1, f);
    fread(&reps, sizeof(reps), 1, f);
    for (int j = 0; j < reps; j++, i++) {
      labels[i] = curr;
    }
  }

  fclose(f);
}

extern "C" void create_graph(long *labels, uint8_t *image, int w, int h,
                             int crop_x, int crop_y, int crop_w, int crop_h,
                             int &num_graphs, int *&sz_edges, long **&edges,
                             int *&sz_features, uint8_t **&features) {

  std::vector<int> crop_labels;
  crop_labels.resize(crop_h * crop_w);

  std::vector<int> node_ids;
  node_ids.resize(crop_h * crop_w);

  std::unordered_map<int, int> id_counter;
  std::unordered_map<int, int> relabel;

  for (int i = 0; i < crop_h; i++) {
    for (int j = 0; j < crop_w; j++) {
      int lbl = labels[(i + crop_y) * w + j + crop_x];
      if (relabel.find(lbl) == relabel.end()) {
        relabel[lbl] = relabel.size();
      }
      lbl = relabel[lbl];
      crop_labels[i * crop_w + j] = lbl;
      node_ids[i * crop_w + j] = id_counter[lbl];
      id_counter[lbl] += 1;
    }
  }

  features = new uint8_t *[id_counter.size() * 3];
  sz_features = new int[id_counter.size()];
  for (int i = 0; i < id_counter.size(); i++) {
    features[i] = new uint8_t[id_counter[i] * 3];
    // use this when counting
    sz_features[i] = 0;
  }

  for (int i = 0; i < crop_h; i++) {
    for (int j = 0; j < crop_w; j++) {
      int lbl = crop_labels[i * crop_w + j];
      int p = ((i + crop_y) * w + j + crop_x) * 3;
      uint8_t *rgb = &features[lbl][sz_features[lbl]];
      rgb[0] = image[p];
      rgb[1] = image[p + 1];
      rgb[2] = image[p + 2];
      sz_features[lbl] += 3;
    }
  }

  num_graphs = id_counter.size();

  edges = new long *[num_graphs];
  sz_edges = new int[num_graphs];
  for (int i = 0; i < num_graphs; i++) {
    // allocate for upper bound of edges
    edges[i] = new long[id_counter[i] * 4];
    // use this when counting
    sz_edges[i] = 0;
  }

  for (int i = 0; i < crop_h; i++) {
    for (int j = 0; j < crop_w; j++) {
      int idx = i * crop_w + j, idx_next = idx + 1, idx_below = idx + crop_w;
      int lbl = crop_labels[idx];
      long *g = edges[lbl];
      int len = sz_edges[lbl];
      if (j != crop_w - 1 && crop_labels[idx_next] == lbl) {
        g[len] = node_ids[idx];
        g[len + 1] = node_ids[idx_next];
        len += 2;
      }
      if (i != crop_h - 1 && crop_labels[idx_below] == lbl) {
        g[len] = node_ids[idx];
        g[len + 1] = node_ids[idx_below];
        len += 2;
      }
      sz_edges[lbl] = len;
    }
  }
}