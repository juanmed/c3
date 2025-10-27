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

    // Add ground with surface velocity
    ProximityProperties ground_props;
    const double point_contact_stiffness = 1.0e4;  // [N/m]
    const double hunt_crossley_dissipation = 1.0;  // [s/m]
    const Eigen::Vector3d velocity_n(1.0, 0.0, 0.0);
    AddContactMaterial(hunt_crossley_dissipation, point_contact_stiffness,
                       CoulombFriction<double>(1.0, 1.0), &ground_props);
    ground_props.AddProperty(kSurfaceVelocityGroup, kSurfaceSpeed, 0.2);
    ground_props.AddProperty(kSurfaceVelocityGroup, kSurfaceVelocityNormal,
                             velocity_n);
    ground_geometry_id_ = plant_->RegisterCollisionGeometry(
        plant_->world_body(),
        drake::math::RigidTransformd(Eigen::Vector3d(0., 0., 0.)), HalfSpace(),
        "ground_collision", ground_props);
    plant_->DeclareSurfaceVelocityInputPort(ground_geometry_id_, Eigen::Vector3d(0., 1., 0.), 1.0);

    // Add sphere on top of ground
    const double radius = 0.05;
    const double mass = 1.0;
    drake::multibody::ModelInstanceIndex sphere_model =
        plant_->AddModelInstance("sphere_instance");
    const auto& sphere_body = plant_->AddRigidBody(
        "sphere_body", sphere_model,
        SpatialInertia<double>::SolidSphereWithMass(mass, radius));
    ProximityProperties sphere_props;
    AddContactMaterial(hunt_crossley_dissipation, point_contact_stiffness,
                       CoulombFriction<double>(0.9, 0.8), &sphere_props);
    sphere_geometry_id_ = plant_->RegisterCollisionGeometry(
        sphere_body, RigidTransform<double>(), Sphere(radius),
        "sphere_collision", sphere_props);

    const Eigen::Vector3d belt_axis = Eigen::Vector3d::UnitX();
    auto& belt_joint = plant_->AddJoint<drake::multibody::PrismaticJoint>(
        "sphere_slider", plant_->world_body(), std::nullopt, sphere_body,
        std::nullopt, belt_axis);
    plant_->AddJointActuator("sphere_actuator", belt_joint);

    plant_->Finalize();

    diagram_ = builder_.Build();
    diagram_context_ = diagram_->CreateDefaultContext();
    plant_context_ =
        &diagram_->GetMutableSubsystemContext(*plant_, diagram_context_.get());

    plant_autodiff_ = System<double>::ToAutoDiffXd(*plant_);
    plant_autodiff_context_ = plant_autodiff_->CreateDefaultContext();

    contact_geometries_.emplace_back(sphere_geometry_id_, ground_geometry_id_);

    options_.contact_model = "stewart_and_trinkle";
    options_.num_contacts = 1;
    options_.num_friction_directions =
        2;  // Square approximation to cone friction
    options_.spring_stiffness = 1.0;
    options_.mu = {0.5};
    options_.N = 1;
    options_.dt = 0.01;

    drake::VectorX<double> state =
        VectorXd::Zero(plant_->num_positions() + plant_->num_velocities());
    drake::VectorX<double> input = VectorXd::Zero(plant_->num_actuators());

    lcs_factory_ = std::make_unique<LCSFactory>(
        *plant_, *plant_context_, *plant_autodiff_, *plant_autodiff_context_,
        contact_geometries_, options_);
    lcs_factory_->UpdateStateAndInput(state, input);
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
  std::unique_ptr<LCSFactory> lcs_factory_;
  drake::geometry::GeometryId ground_geometry_id_;
  drake::geometry::GeometryId sphere_geometry_id_;
};

}  // namespace test
}  // namespace multibody
}  // namespace c3
