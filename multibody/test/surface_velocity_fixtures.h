#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <optional>

#include "multibody/lcs_factory.h"
#include "multibody/lcs_factory_options.h"

#include "drake/common/autodiff.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/system.h"

namespace c3 {
namespace multibody {
namespace test {

using drake::geometry::AddContactMaterial;
using drake::geometry::GeometryId;
using drake::geometry::HalfSpace;
using drake::geometry::ProximityProperties;
using drake::geometry::SceneGraph;
using drake::geometry::Sphere;
using drake::geometry::internal::kSurfaceSpeed;
using drake::geometry::internal::kSurfaceVelocityGroup;
using drake::geometry::internal::kSurfaceVelocityNormal;
using drake::math::RigidTransform;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::CoulombFriction;
using drake::multibody::MultibodyPlant;
using drake::multibody::SpatialInertia;
using drake::systems::Context;
using drake::systems::Diagram;
using drake::systems::DiagramBuilder;
using drake::systems::System;

class SurfaceVelocityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::tie(plant_, scene_graph_) =
        AddMultibodyPlantSceneGraph(&builder_, 0.0);

    drake::multibody::Parser parser(plant_, scene_graph_);
    parser.AddModels("examples/resources/conveyor_belt/conveyor_belt.sdf");

    // Add an input port for surface velocity
    const drake::multibody::RigidBody<double>& conveyor_belt_body =
        plant_->GetBodyByName("conveyor_belt");
    const drake::geometry::GeometryId geom_id =
        plant_->GetCollisionGeometriesForBody(conveyor_belt_body).at(0);
    plant_->DeclareSurfaceVelocityInputPort(
        geom_id, Eigen::Vector3d(0.0, 1.0, 0.0), 0.0);

    plant_->set_name("plant_");
    plant_->Finalize();

    diagram_ = builder_.Build();
    diagram_context_ = diagram_->CreateDefaultContext();
    plant_context_ =
        &diagram_->GetMutableSubsystemContext(*plant_, diagram_context_.get());

    plant_autodiff_ = System<double>::ToAutoDiffXd(*plant_);
    plant_autodiff_context_ = plant_autodiff_->CreateDefaultContext();

    // Retrieve collision geometries for relevant bodies.
    std::vector<GeometryId> conveyor_belt_collision_geoms =
        plant_->GetCollisionGeometriesForBody(
            plant_->GetBodyByName("conveyor_belt"));
    std::vector<GeometryId> sphere_collision_geoms =
        plant_->GetCollisionGeometriesForBody(plant_->GetBodyByName("sphere"));

    conveyor_belt_geometry_id_ = conveyor_belt_collision_geoms[0];
    sphere_geometry_id_ = sphere_collision_geoms[0];
    contact_geometries_.emplace_back(conveyor_belt_geometry_id_,
                                     sphere_geometry_id_);

    options_.contact_model = "stewart_and_trinkle";
    options_.num_contacts = 1;
    options_.num_friction_directions =
        2;  // Square approximation to cone friction
    options_.spring_stiffness = 1.0;
    options_.mu = {0.5};
    options_.N = 1;
    options_.dt = 0.01;

    // Create some state and input vectors to update the LCS
    // Make sure to not zero all elements of state because some correspond
    // to orientation, which an throw if an ill-formed element is passed
    const auto q0 = plant_->GetPositions(*plant_context_);
    const auto v0 = plant_->GetVelocities(*plant_context_);
    drake::VectorX<double> state(q0.size() + v0.size());
    state << q0, v0;
    drake::VectorX<double> input = VectorXd::Zero(plant_->num_actuators());
  }

  DiagramBuilder<double> builder_;
  MultibodyPlant<double>* plant_{nullptr};
  SceneGraph<double>* scene_graph_{nullptr};
  std::unique_ptr<Diagram<double>> diagram_;
  std::unique_ptr<Context<double>> diagram_context_;
  Context<double>* plant_context_{nullptr};
  std::unique_ptr<MultibodyPlant<drake::AutoDiffXd>> plant_autodiff_;
  std::unique_ptr<Context<drake::AutoDiffXd>> plant_autodiff_context_;
  LCSFactoryOptions options_;
  std::vector<drake::SortedPair<drake::geometry::GeometryId>>
      contact_geometries_;
  drake::geometry::GeometryId conveyor_belt_geometry_id_;
  drake::geometry::GeometryId sphere_geometry_id_;
};

}  // namespace test
}  // namespace multibody
}  // namespace c3
