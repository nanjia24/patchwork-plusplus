#include "../src/GroundSurfaceRecheck.hpp"
#include <iostream>
using namespace patchworkpp_ros;
void require(bool v,const char *s){if(!v)throw std::runtime_error(s);}
int main(){
 GroundSurfaceRecheck c;
 Eigen::MatrixX3f g(9,3),o(6,3);
 for(int i=0;i<9;++i)g.row(i)<<2.+(i%3)*.02,(i/3)*.02,-.16;
 o<<2.65,0.,-.165, 2.65,0.,-.09, 2.65,0.,.0, 4.,0.,-.16, 6.,0.,-.16, 2.65,0.,-.21;
 const auto old=o;require(c.apply(g,o)==1,"only coplanar measured point corrected");
 require(g.rows()==10 && o.rows()==5,"point count preserved");
 require((g.row(9)-old.row(0)).norm()==0,"no synthesized point");
 require((o-old.bottomRows(5)).norm()==0,"7cm obstacle, wall, unsupported, out of range, lower step preserved");
 g=Eigen::MatrixX3f::Zero(8,3);g.col(0).setConstant(2.);g.col(2).setConstant(-.16);
 o.resize(1,3);o<<2.1,0.,-.16;require(c.apply(g,o)==0,"duplicates cannot supply support");
 g.resize(9,3);for(int i=0;i<9;++i)g.row(i)<<-2.-(i%3)*.02,-(i/3)*.02,-.16;
 o<<-2.65,0.,-.16;require(c.apply(g,o)==1,"negative hash coordinates work");
 g.resize(0,3);require(c.apply(g,o)==0,"empty safe");
 std::cout<<"surface correction synthetic checks passed\n";
}
