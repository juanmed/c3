/* This file was fully generated using openAI gpt-5-codex-medium*/

#include "multibody/multibody_utils.h"

#include <gtest/gtest.h>

#include <optional>
#include <set>

#include <Eigen/Core>

#include "drake/common/autodiff.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/multibody/tree/unit_inertia.h"
#include "drake/systems/framework/system.h"

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::VectorX;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryId;
using drake::geometry::ProximityProperties;
using drake::geometry::SceneGraph;
using drake::geometry::internal::kSurfaceSpeed;
using drake::geometry::internal::kSurfaceVelocityGroup;
using drake::geometry::internal::kSurfaceVelocityNormal;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;
using drake::math::InitializeAutoDiff;
using drake::math::RigidTransform;
using drake::multibody::CoulombFriction;
using drake::multibody::MultibodyPlant;
using drake::multibody::SpatialInertia;
using drake::multibody::UnitInertia;
using drake::systems::Context;
using drake::systems::System;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace c3 {
namespace multibody {
namespace test {

namespace {

VectorXd MakeVector(int size, double start, double step) {
  VectorXd values(size);
  for (int i = 0; i < size; ++i) {
    values[i] = start + step * i;
  }
  return values;
}

AutoDiffVecXd MakeAutoDiffVector(int size, double start, double step) {
  return InitializeAutoDiff(MakeVector(size, start, step));
}

void ExpectAutoDiffEqual(const AutoDiffVecXd& actual,
                         const AutoDiffVecXd& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_TRUE(ExtractValue(actual).isApprox(ExtractValue(expected)));
  EXPECT_TRUE(ExtractGradient(actual).isApprox(ExtractGradient(expected)));
}

class ConveyorBeltPlantFixtureBase {
 protected:
  void Initialize(bool enable_surface_velocity) {
    plant_ = std::make_unique<MultibodyPlant<double>>(0.0);
    scene_graph_ = std::make_unique<SceneGraph<double>>();
    plant_->RegisterAsSourceForSceneGraph(scene_graph_.get());

    const double belt_mass = 5.0;
    const double box_mass = 2.5;
    const double belt_length = 1.2;
    const double belt_width = 0.5;
    const double belt_height = 0.1;
    const double box_edge = 0.3;

    const auto belt_instance = plant_->AddModelInstance("conveyor_belt");
    const auto box_instance = plant_->AddModelInstance("box");

    const SpatialInertia<double> belt_inertia =
        SpatialInertia<double>::MakeFromCentralInertia(
            belt_mass, Vector3d::Zero(),
            UnitInertia<double>::SolidBox(belt_length, belt_width,
                                          belt_height));
    const SpatialInertia<double> box_inertia =
        SpatialInertia<double>::MakeFromCentralInertia(
            box_mass, Vector3d::Zero(),
            UnitInertia<double>::SolidBox(box_edge, box_edge, box_edge));

    const auto& belt_body =
        plant_->AddRigidBody("belt_body", belt_instance, belt_inertia);
    const auto& box_body =
        plant_->AddRigidBody("box_body", box_instance, box_inertia);

    const Vector3d belt_axis = Vector3d::UnitX();
    auto& belt_joint = plant_->AddJoint<drake::multibody::PrismaticJoint>(
        "belt_slider", plant_->world_body(), std::nullopt, belt_body,
        std::nullopt, belt_axis);
    plant_->AddJointActuator("belt_actuator", belt_joint);

    const Vector3d lift_axis = Vector3d::UnitZ();
    auto& box_joint = plant_->AddJoint<drake::multibody::PrismaticJoint>(
        "box_elevator", plant_->world_body(), std::nullopt, box_body,
        std::nullopt, lift_axis);
    plant_->AddJointActuator("box_actuator", box_joint);

    ProximityProperties belt_collision;
    const double dissipation = 1.0;
    const double stiffness = 1.0e4;
    AddContactMaterial(dissipation, stiffness,
                       CoulombFriction<double>(0.9, 0.8), &belt_collision);
    if (enable_surface_velocity) {
      belt_collision.AddProperty(kSurfaceVelocityGroup, kSurfaceSpeed, 1.25);
      belt_collision.AddProperty(kSurfaceVelocityGroup, kSurfaceVelocityNormal,
                                 belt_axis);
    }
    belt_geometry_id_ = plant_->RegisterCollisionGeometry(
        belt_body, RigidTransform<double>(), Box(belt_length, belt_width,
                                                 belt_height),
        "belt_collision", belt_collision);

    if (enable_surface_velocity) {
      plant_->DeclareSurfaceVelocityInputPort(belt_geometry_id_, belt_axis,
                                              1.25);
    }

    ProximityProperties box_collision;
    AddContactMaterial(dissipation, stiffness,
                       CoulombFriction<double>(0.7, 0.6), &box_collision);
    box_geometry_id_ = plant_->RegisterCollisionGeometry(
        box_body, RigidTransform<double>(Vector3d(0., 0., box_edge / 2.0)),
        Box(box_edge, box_edge, box_edge), "box_collision", box_collision);

    plant_->Finalize();

    context_ = plant_->CreateDefaultContext();
    plant_autodiff_ = System<double>::ToAutoDiffXd(*plant_);
    context_autodiff_ = plant_autodiff_->CreateDefaultContext();
    has_surface_velocity_inputs_ =
        plant_->get_surface_speed_input_port(belt_geometry_id_).has_value();

    const auto& belt_body_ad =
        plant_autodiff_->GetBodyByName("belt_body");
    const auto& belt_geoms_ad =
        plant_autodiff_->GetCollisionGeometriesForBody(belt_body_ad);
    belt_geometry_id_autodiff_ = belt_geoms_ad.at(0);
    has_surface_velocity_autodiff_inputs_ =
        plant_autodiff_->get_surface_speed_input_port(
            belt_geometry_id_autodiff_)
            .has_value();
    surface_velocity_geometries_.clear();
    surface_velocity_geometries_.insert(belt_geometry_id_);
    surface_velocity_geometries_autodiff_.clear();
    surface_velocity_geometries_autodiff_.insert(belt_geometry_id_autodiff_);
  }

  MultibodyPlant<double>& plant_double() { return *plant_; }
  const MultibodyPlant<double>& plant_double() const { return *plant_; }
  Context<double>* mutable_context() { return context_.get(); }
  const Context<double>& context() const { return *context_; }

  MultibodyPlant<AutoDiffXd>& plant_autodiff() { return *plant_autodiff_; }
  const MultibodyPlant<AutoDiffXd>& plant_autodiff() const {
    return *plant_autodiff_;
  }
  Context<AutoDiffXd>* mutable_context_autodiff() {
    return context_autodiff_.get();
  }
  const Context<AutoDiffXd>& context_autodiff() const {
    return *context_autodiff_;
  }

  GeometryId belt_geometry_id() const { return belt_geometry_id_; }
  GeometryId belt_geometry_id_autodiff() const { return belt_geometry_id_autodiff_; }
  const std::set<GeometryId>& surface_velocity_geometries() const {
    return surface_velocity_geometries_;
  }
  const std::set<GeometryId>& surface_velocity_geometries_autodiff() const {
    return surface_velocity_geometries_autodiff_;
  }
  bool supports_surface_velocity() const {
    return has_surface_velocity_inputs_;
  }
  bool supports_surface_velocity_autodiff() const {
    return has_surface_velocity_autodiff_inputs_;
  }

 private:
  std::unique_ptr<MultibodyPlant<double>> plant_;
  std::unique_ptr<SceneGraph<double>> scene_graph_;
  std::unique_ptr<Context<double>> context_;
  std::unique_ptr<MultibodyPlant<AutoDiffXd>> plant_autodiff_;
  std::unique_ptr<Context<AutoDiffXd>> context_autodiff_;
  GeometryId belt_geometry_id_;
  GeometryId box_geometry_id_;
  GeometryId belt_geometry_id_autodiff_;
  std::set<GeometryId> surface_velocity_geometries_;
  std::set<GeometryId> surface_velocity_geometries_autodiff_;
  bool has_surface_velocity_inputs_{false};
  bool has_surface_velocity_autodiff_inputs_{false};
};

class ConveyorBeltWithoutSurfaceVelocityTest : public ::testing::Test,
                                                protected ConveyorBeltPlantFixtureBase {
 protected:
  void SetUp() override { Initialize(false); }
};

class ConveyorBeltWithSurfaceVelocityTest : public ::testing::Test,
                                             protected ConveyorBeltPlantFixtureBase {
 protected:
  void SetUp() override { Initialize(true); }
};

template <typename Fixture>
class MultibodyUtilsTest : public Fixture {};

using TestFixtures =
    ::testing::Types<ConveyorBeltWithoutSurfaceVelocityTest,
                     ConveyorBeltWithSurfaceVelocityTest>;
TYPED_TEST_SUITE(MultibodyUtilsTest, TestFixtures);

TYPED_TEST(MultibodyUtilsTest, SetPositionsIfNewUpdatesContext) {
  auto& plant_double = this->plant_double();
  Context<double>* context_double = this->mutable_context();
  const VectorXd q = MakeVector(plant_double.num_positions(), 0.1, 0.05);
  SetPositionsIfNew<double>(plant_double, q, context_double);
  EXPECT_TRUE(plant_double.GetPositions(*context_double).isApprox(q));

  const VectorXd q_new =
      MakeVector(plant_double.num_positions(), -0.2, 0.07);
  SetPositionsIfNew<double>(plant_double, q_new, context_double);
  EXPECT_TRUE(plant_double.GetPositions(*context_double).isApprox(q_new));

  auto& plant_autodiff = this->plant_autodiff();
  Context<AutoDiffXd>* context_autodiff = this->mutable_context_autodiff();
  const AutoDiffVecXd q_ad =
      MakeAutoDiffVector(plant_autodiff.num_positions(), 0.1, 0.05);
  SetPositionsIfNew<AutoDiffXd>(plant_autodiff, q_ad, context_autodiff);
  ExpectAutoDiffEqual(plant_autodiff.GetPositions(*context_autodiff), q_ad);

  const AutoDiffVecXd q_ad_new =
      MakeAutoDiffVector(plant_autodiff.num_positions(), -0.2, 0.07);
  SetPositionsIfNew<AutoDiffXd>(plant_autodiff, q_ad_new, context_autodiff);
  ExpectAutoDiffEqual(plant_autodiff.GetPositions(*context_autodiff),
                      q_ad_new);
}

TYPED_TEST(MultibodyUtilsTest, SetVelocitiesIfNewUpdatesContext) {
  auto& plant_double = this->plant_double();
  Context<double>* context_double = this->mutable_context();
  const VectorXd v = MakeVector(plant_double.num_velocities(), 0.3, -0.04);
  SetVelocitiesIfNew<double>(plant_double, v, context_double);
  EXPECT_TRUE(plant_double.GetVelocities(*context_double).isApprox(v));

  const VectorXd v_new =
      MakeVector(plant_double.num_velocities(), -0.1, 0.06);
  SetVelocitiesIfNew<double>(plant_double, v_new, context_double);
  EXPECT_TRUE(plant_double.GetVelocities(*context_double).isApprox(v_new));

  auto& plant_autodiff = this->plant_autodiff();
  Context<AutoDiffXd>* context_autodiff = this->mutable_context_autodiff();
  const AutoDiffVecXd v_ad =
      MakeAutoDiffVector(plant_autodiff.num_velocities(), 0.3, -0.04);
  SetVelocitiesIfNew<AutoDiffXd>(plant_autodiff, v_ad, context_autodiff);
  ExpectAutoDiffEqual(plant_autodiff.GetVelocities(*context_autodiff), v_ad);

  const AutoDiffVecXd v_ad_new =
      MakeAutoDiffVector(plant_autodiff.num_velocities(), -0.1, 0.06);
  SetVelocitiesIfNew<AutoDiffXd>(plant_autodiff, v_ad_new, context_autodiff);
  ExpectAutoDiffEqual(plant_autodiff.GetVelocities(*context_autodiff),
                      v_ad_new);
}

TYPED_TEST(MultibodyUtilsTest, SetInputsIfNewUpdatesContext) {
  auto& plant_double = this->plant_double();
  Context<double>* context_double = this->mutable_context();
  const VectorXd u =
      MakeVector(plant_double.num_actuators(), 0.5, -0.08);
  SetInputsIfNew<double>(plant_double, u, context_double);
  EXPECT_TRUE(
      plant_double.get_actuation_input_port().Eval(*context_double).isApprox(
          u));

  const VectorXd u_new =
      MakeVector(plant_double.num_actuators(), -0.3, 0.11);
  SetInputsIfNew<double>(plant_double, u_new, context_double);
  EXPECT_TRUE(
      plant_double.get_actuation_input_port().Eval(*context_double).isApprox(
          u_new));

  auto& plant_autodiff = this->plant_autodiff();
  Context<AutoDiffXd>* context_autodiff = this->mutable_context_autodiff();
  const AutoDiffVecXd u_ad =
      MakeAutoDiffVector(plant_autodiff.num_actuators(), 0.5, -0.08);
  SetInputsIfNew<AutoDiffXd>(plant_autodiff, u_ad, context_autodiff);
  ExpectAutoDiffEqual(
      plant_autodiff.get_actuation_input_port().Eval(*context_autodiff),
      u_ad);

  const AutoDiffVecXd u_ad_new =
      MakeAutoDiffVector(plant_autodiff.num_actuators(), -0.3, 0.11);
  SetInputsIfNew<AutoDiffXd>(plant_autodiff, u_ad_new, context_autodiff);
  ExpectAutoDiffEqual(
      plant_autodiff.get_actuation_input_port().Eval(*context_autodiff),
      u_ad_new);
}

TYPED_TEST(MultibodyUtilsTest, SetPositionsAndVelocitiesIfNewUpdatesContext) {
  auto& plant_double = this->plant_double();
  Context<double>* context_double = this->mutable_context();
  const VectorXd q =
      MakeVector(plant_double.num_positions(), 0.12, 0.03);
  const VectorXd v =
      MakeVector(plant_double.num_velocities(), -0.09, 0.02);
  VectorXd state(plant_double.num_positions() + plant_double.num_velocities());
  state << q, v;
  SetPositionsAndVelocitiesIfNew<double>(plant_double, state,
                                         context_double);
  EXPECT_TRUE(plant_double.GetPositions(*context_double).isApprox(q));
  EXPECT_TRUE(plant_double.GetVelocities(*context_double).isApprox(v));

  const VectorXd q_new =
      MakeVector(plant_double.num_positions(), -0.25, 0.04);
  const VectorXd v_new =
      MakeVector(plant_double.num_velocities(), 0.15, -0.05);
  VectorXd state_new(
      plant_double.num_positions() + plant_double.num_velocities());
  state_new << q_new, v_new;
  SetPositionsAndVelocitiesIfNew<double>(plant_double, state_new,
                                         context_double);
  EXPECT_TRUE(plant_double.GetPositions(*context_double).isApprox(q_new));
  EXPECT_TRUE(plant_double.GetVelocities(*context_double).isApprox(v_new));

  auto& plant_autodiff = this->plant_autodiff();
  Context<AutoDiffXd>* context_autodiff = this->mutable_context_autodiff();
  const AutoDiffVecXd state_ad =
      MakeAutoDiffVector(plant_autodiff.num_positions() +
                             plant_autodiff.num_velocities(),
                         0.12, 0.03);
  SetPositionsAndVelocitiesIfNew<AutoDiffXd>(plant_autodiff, state_ad,
                                             context_autodiff);
  ExpectAutoDiffEqual(
      plant_autodiff.GetPositions(*context_autodiff),
      state_ad.head(plant_autodiff.num_positions()));
  ExpectAutoDiffEqual(
      plant_autodiff.GetVelocities(*context_autodiff),
      state_ad.tail(plant_autodiff.num_velocities()));

  const AutoDiffVecXd state_ad_new =
      MakeAutoDiffVector(plant_autodiff.num_positions() +
                             plant_autodiff.num_velocities(),
                         -0.25, 0.04);
  SetPositionsAndVelocitiesIfNew<AutoDiffXd>(plant_autodiff, state_ad_new,
                                             context_autodiff);
  ExpectAutoDiffEqual(
      plant_autodiff.GetPositions(*context_autodiff),
      state_ad_new.head(plant_autodiff.num_positions()));
  ExpectAutoDiffEqual(
      plant_autodiff.GetVelocities(*context_autodiff),
      state_ad_new.tail(plant_autodiff.num_velocities()));
}

TYPED_TEST(MultibodyUtilsTest, SetContextUpdatesStateAndInput) {
  auto& plant_double = this->plant_double();
  Context<double>* context_double = this->mutable_context();
  const VectorXd q =
      MakeVector(plant_double.num_positions(), 0.21, -0.03);
  const VectorXd v =
      MakeVector(plant_double.num_velocities(), -0.14, 0.02);
  VectorXd state(plant_double.num_positions() + plant_double.num_velocities());
  state << q, v;
  const VectorXd u =
      MakeVector(plant_double.num_actuators(), 0.6, -0.1);
  SetContext<double>(plant_double, state, u, context_double);
  EXPECT_TRUE(plant_double.GetPositions(*context_double).isApprox(q));
  EXPECT_TRUE(plant_double.GetVelocities(*context_double).isApprox(v));
  EXPECT_TRUE(
      plant_double.get_actuation_input_port().Eval(*context_double).isApprox(
          u));

  auto& plant_autodiff = this->plant_autodiff();
  Context<AutoDiffXd>* context_autodiff = this->mutable_context_autodiff();
  const AutoDiffVecXd state_ad =
      MakeAutoDiffVector(plant_autodiff.num_positions() +
                             plant_autodiff.num_velocities(),
                         0.21, -0.03);
  const AutoDiffVecXd u_ad =
      MakeAutoDiffVector(plant_autodiff.num_actuators(), 0.6, -0.1);
  SetContext<AutoDiffXd>(plant_autodiff, state_ad, u_ad, context_autodiff);
  ExpectAutoDiffEqual(
      plant_autodiff.GetPositions(*context_autodiff),
      state_ad.head(plant_autodiff.num_positions()));
  ExpectAutoDiffEqual(
      plant_autodiff.GetVelocities(*context_autodiff),
      state_ad.tail(plant_autodiff.num_velocities()));
  ExpectAutoDiffEqual(
      plant_autodiff.get_actuation_input_port().Eval(*context_autodiff),
      u_ad);
}

TYPED_TEST(MultibodyUtilsTest, SetContextWithSurfaceVelocitiesUpdatesAllInputs) {
  auto& plant_double = this->plant_double();
  Context<double>* context_double = this->mutable_context();
  const VectorXd q =
      MakeVector(plant_double.num_positions(), 0.05, 0.04);
  const VectorXd v =
      MakeVector(plant_double.num_velocities(), -0.08, 0.03);
  VectorXd state(plant_double.num_positions() + plant_double.num_velocities());
  state << q, v;
  const VectorXd u =
      MakeVector(plant_double.num_actuators(), -0.2, 0.05);
  const VectorXd constraint_inputs =
      MakeVector(static_cast<int>(this->surface_velocity_geometries().size()),
                 1.1, -0.2);

  SetContext<double>(plant_double, state, u, constraint_inputs,
                     this->surface_velocity_geometries(), context_double);

  EXPECT_TRUE(plant_double.GetPositions(*context_double).isApprox(q));
  EXPECT_TRUE(plant_double.GetVelocities(*context_double).isApprox(v));
  EXPECT_TRUE(
      plant_double.get_actuation_input_port().Eval(*context_double).isApprox(
          u));

  const auto surface_port =
      plant_double.get_surface_speed_input_port(this->belt_geometry_id());
  if (this->supports_surface_velocity()) {
    ASSERT_TRUE(surface_port.has_value());
    VectorXd port_value =
        surface_port.value().get().Eval(*context_double);
    EXPECT_EQ(port_value.size(), constraint_inputs.size());
    EXPECT_TRUE(port_value.isApprox(constraint_inputs));
  } else {
    EXPECT_FALSE(surface_port.has_value());
  }

  auto& plant_autodiff = this->plant_autodiff();
  Context<AutoDiffXd>* context_autodiff = this->mutable_context_autodiff();
  const AutoDiffVecXd state_ad =
      MakeAutoDiffVector(plant_autodiff.num_positions() +
                             plant_autodiff.num_velocities(),
                         0.05, 0.04);
  const AutoDiffVecXd u_ad =
      MakeAutoDiffVector(plant_autodiff.num_actuators(), -0.2, 0.05);
  const AutoDiffVecXd constraint_inputs_ad = MakeAutoDiffVector(
      static_cast<int>(this->surface_velocity_geometries_autodiff().size()),
      1.1, -0.2);

  SetContext<AutoDiffXd>(plant_autodiff, state_ad, u_ad, constraint_inputs_ad,
                         this->surface_velocity_geometries_autodiff(),
                         context_autodiff);

  ExpectAutoDiffEqual(
      plant_autodiff.GetPositions(*context_autodiff),
      state_ad.head(plant_autodiff.num_positions()));
  ExpectAutoDiffEqual(
      plant_autodiff.GetVelocities(*context_autodiff),
      state_ad.tail(plant_autodiff.num_velocities()));
  ExpectAutoDiffEqual(
      plant_autodiff.get_actuation_input_port().Eval(*context_autodiff),
      u_ad);

  const auto surface_port_ad =
      plant_autodiff.get_surface_speed_input_port(
          this->belt_geometry_id_autodiff());
  if (this->supports_surface_velocity_autodiff()) {
    ASSERT_TRUE(surface_port_ad.has_value());
    AutoDiffVecXd port_value =
        surface_port_ad.value().get().Eval(*context_autodiff);
    ExpectAutoDiffEqual(port_value, constraint_inputs_ad);
  } else {
    EXPECT_FALSE(surface_port_ad.has_value());
  }
}

TYPED_TEST(MultibodyUtilsTest, SetSurfaceVelocitiesIfNewHandlesGeometries) {
  auto& plant_double = this->plant_double();
  Context<double>* context_double = this->mutable_context();
  const VectorXd constraint_inputs =
      MakeVector(static_cast<int>(this->surface_velocity_geometries().size()),
                 0.75, -0.15);
  SetSurfaceVelocitiesIfNew<double>(plant_double, constraint_inputs,
                                    this->surface_velocity_geometries(),
                                    context_double);

  const auto surface_port =
      plant_double.get_surface_speed_input_port(this->belt_geometry_id());
  if (this->supports_surface_velocity()) {
    ASSERT_TRUE(surface_port.has_value());
    VectorXd port_value =
        surface_port.value().get().Eval(*context_double);
    EXPECT_TRUE(port_value.isApprox(constraint_inputs));
  } else {
    EXPECT_FALSE(surface_port.has_value());
  }

  auto& plant_autodiff = this->plant_autodiff();
  Context<AutoDiffXd>* context_autodiff = this->mutable_context_autodiff();
  const AutoDiffVecXd constraint_inputs_ad = MakeAutoDiffVector(
      static_cast<int>(this->surface_velocity_geometries_autodiff().size()),
      0.75, -0.15);
  SetSurfaceVelocitiesIfNew<AutoDiffXd>(
      plant_autodiff, constraint_inputs_ad,
      this->surface_velocity_geometries_autodiff(), context_autodiff);

  const auto surface_port_ad =
      plant_autodiff.get_surface_speed_input_port(
          this->belt_geometry_id_autodiff());
  if (this->supports_surface_velocity_autodiff()) {
    ASSERT_TRUE(surface_port_ad.has_value());
    AutoDiffVecXd port_value =
        surface_port_ad.value().get().Eval(*context_autodiff);
    ExpectAutoDiffEqual(port_value, constraint_inputs_ad);
  } else {
    EXPECT_FALSE(surface_port_ad.has_value());
  }
}

}  // namespace
}  // namespace test
}  // namespace multibody
}  // namespace c3
