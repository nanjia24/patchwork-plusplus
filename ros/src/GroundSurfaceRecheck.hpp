#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace patchworkpp_ros {
// Same-frame, measured-point correction only. No fitted/extrapolated plane,
// synthetic returns, history, or recursive use of corrected points as support.
struct GroundSurfaceRecheck {
  double range = 5.0;
  double radius = 1.0;
  double height = 0.02;
  int neighbors = 5;
  void validate() const {
    if (!std::isfinite(range) || !std::isfinite(radius) || !std::isfinite(height) ||
        range <= 0 || radius <= 0 || height <= 0 || neighbors < 3 || neighbors > 32)
      throw std::invalid_argument("Invalid ground surface recheck parameters");
  }
  static uint64_t key(int x, int y) {
    return (uint64_t(uint32_t(x)) << 32) | uint32_t(y);
  }
  size_t apply(Eigen::MatrixX3f &ground, Eigen::MatrixX3f &obstacle) const {
    validate();
    if (ground.rows() < neighbors || obstacle.rows() == 0) return 0;
    std::vector<int> candidates;
    for (int i=0; i<obstacle.rows(); ++i)
      if (obstacle.row(i).allFinite() && obstacle.row(i).head<2>().squaredNorm() <= range*range)
        candidates.push_back(i);
    if (candidates.empty()) return 0;
    // Distinct 1 cm XY cells prevent repeated/oversampled returns supplying all
    // votes. Keep the whole cell's height interval to reject ambiguous levels.
    struct Support { double x,y,low,high; };
    std::unordered_map<uint64_t, Support> unique;
    for (int i=0; i<ground.rows(); ++i) {
      if (!ground.row(i).allFinite() || ground.row(i).head<2>().squaredNorm() > (range+radius)*(range+radius)) continue;
      const double x=ground(i,0), y=ground(i,1), z=ground(i,2);
      const int ix=int(std::floor(x/.01)), iy=int(std::floor(y/.01));
      auto [it, inserted]=unique.emplace(key(ix,iy), Support{(ix+.5)*.01,(iy+.5)*.01,z,z});
      if (!inserted) { it->second.low=std::min(it->second.low,z); it->second.high=std::max(it->second.high,z); }
    }
    std::unordered_map<uint64_t,std::vector<Support>> cells;
    for (const auto &item:unique) {
      const auto &p=item.second;
      cells[key(int(std::floor(p.x/radius)),int(std::floor(p.y/radius)))].push_back(p);
    }
    std::vector<char> corrected(obstacle.rows(),0);
    size_t count=0;
    std::vector<std::pair<double, const Support*>> closest;
    closest.reserve(neighbors);
    for (int i:candidates) {
      closest.clear();
      const double x=obstacle(i,0),y=obstacle(i,1),z=obstacle(i,2);
      const int cx=int(std::floor(x/radius)), cy=int(std::floor(y/radius));
      for (int dx=-1;dx<=1;++dx) for (int dy=-1;dy<=1;++dy) {
        auto it=cells.find(key(cx+dx,cy+dy)); if(it==cells.end()) continue;
        for(const auto &p:it->second) {
          const double d=(p.x-x)*(p.x-x)+(p.y-y)*(p.y-y);
          if(d>radius*radius || (int(closest.size())==neighbors && d>=closest.back().first)) continue;
          auto pos=std::lower_bound(closest.begin(),closest.end(),d,
              [](const auto &entry,double distance){return entry.first<distance;});
          closest.insert(pos,{d,&p});if(int(closest.size())>neighbors)closest.pop_back();
        }
      }
      if(int(closest.size())!=neighbors)continue;
      bool supported=true;
      for(const auto &p:closest)
        if(std::abs(p.second->low-z)>height || std::abs(p.second->high-z)>height) { supported=false; break; }
      if(supported) { corrected[i]=1; ++count; }
    }
    if(!count)return 0;
    const auto old=ground.rows();ground.conservativeResize(old+count,Eigen::NoChange);
    Eigen::Index g=old,o=0;
    for(Eigen::Index i=0;i<obstacle.rows();++i) {
      if(corrected[i])ground.row(g++)=obstacle.row(i);
      else { if(o!=i)obstacle.row(o)=obstacle.row(i); ++o; }
    }
    obstacle.conservativeResize(o,Eigen::NoChange);
    return count;
  }
};
}
