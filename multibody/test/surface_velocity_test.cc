#include <gtest/gtest.h>

#include "surface_velocity_fixtures.h"

namespace c3 {
namespace multibody {
namespace test {

TEST_F(SurfaceVelocityTest, GetNumContactVelocityBiases) {
  int n_b = lcs_factory_->GetNumContactVelocityBiases(*plant_, *plant_context_,
                                                      contact_geometries_);
  EXPECT_EQ(n_b, 1);
}

}  // namespace test
}  // namespace multibody
}  // namespace c3
