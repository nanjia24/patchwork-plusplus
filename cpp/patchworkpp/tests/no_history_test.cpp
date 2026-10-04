#include "patchwork/patchworkpp.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>

using Cloud = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
static std::vector<int> labels(patchwork::PatchWorkpp& alg, const Cloud& cloud) {
  alg.estimateGround(cloud);
  auto indices = alg.getNongroundIndices();
  std::vector<int> result(indices.data(), indices.data() + indices.size());
  std::sort(result.begin(), result.end());
  return result;
}
static void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
int main(int argc, char** argv) {
  patchwork::Params p;
  require(p.enable_adaptive_learning, "Other users must retain learning by default");
  p.enable_adaptive_learning=false;p.enable_RNR=false;
  p.sensor_height=.16;p.min_range=.1;p.max_range=40;p.th_dist=.05;
  patchwork::PatchWorkpp history(p);
  Cloud floor(6000,3);
  for(int i=0;i<6000;i++) floor.row(i)<<.2+(i%100)*.025, -1.2+(i/100)*.04, -.16;
  Cloud lower=floor;lower.col(2).array()-=.05;
  for(int i=0;i<100;i++) history.estimateGround(lower);
  patchwork::PatchWorkpp fresh(p);
  require(labels(history,floor)==labels(fresh,floor), "Earlier height must not affect current classification");
  // A 20 cm tall obstacle remains above the 5 cm local plane-distance cutoff.
  Cloud obstacle=floor;
  int count=0;
  for(int i=0;i<6000;i++) if(floor(i,0)>1. && floor(i,0)<1.2 && std::abs(floor(i,1))<.15) {
    obstacle(i,2)+=.2;count++;
  }
  auto out=labels(history,obstacle);int retained=0;
  for(auto i:out) if(obstacle(i,2)>0)retained++;
  require(count>0 && retained>=count*.95, "20 cm obstacle must be retained");
  // Optional recorded raw stream: every frame must equal a fresh classifier,
  // including after the artificial historical floor-height shift above.
  if(argc>1) {
    std::ifstream f(argv[1],std::ios::binary);require(bool(f),"raw stream missing");
    uint32_t n;int frames=0;
    while(f.read(reinterpret_cast<char*>(&n),4)) {
      Cloud a(n,3);require(bool(f.read(reinterpret_cast<char*>(a.data()),n*12)),"truncated frame");
      patchwork::PatchWorkpp independent(p);
      require(labels(history,a)==labels(independent,a),"Recorded frame depends on history");frames++;
    }
    require(frames>0,"empty stream");std::cout<<"Recorded history-independence frames: "<<frames<<std::endl;
  }
  std::cout<<"No-history tests passed; obstacle retained "<<retained<<"/"<<count<<std::endl;
}
