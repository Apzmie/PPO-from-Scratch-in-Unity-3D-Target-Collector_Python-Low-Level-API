using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class WheelAgent : Agent
{
    public Transform target;
    
    private float spring = 0f;
    private float damper = 100f;
    private float maxForce = 10000f;
    private float maxVelocity = 10f;
    private float targetRange = 8f;
    
    private Rigidbody rBody;
    private ConfigurableJoint[] joints;
    private Rigidbody[] jointRbs;
    private Vector3[] jointStartPos;
    private Quaternion[] jointStartRot;
    private Vector3 startPos;
    private Quaternion startRot;
    
    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();
        joints = GetComponentsInChildren<ConfigurableJoint>();
        jointRbs = new Rigidbody[joints.Length];
        
        jointStartPos = new Vector3[joints.Length];
        jointStartRot = new Quaternion[joints.Length];

        for (int i = 0; i < joints.Length; i++)
        {
            jointRbs[i] = joints[i].GetComponent<Rigidbody>();
            jointStartPos[i] = joints[i].transform.localPosition;
            jointStartRot[i] = joints[i].transform.localRotation;
            
            ConfigurableJoint joint = joints[i];
            JointDrive drive = joint.angularXDrive;
            drive.positionSpring = spring;
            drive.positionDamper = damper;
            drive.maximumForce = maxForce;
            joint.angularXDrive = drive;
        }
        
        startPos = transform.localPosition;
        startRot = transform.localRotation; 
    }
    
    public override void OnEpisodeBegin()
    {
        transform.localPosition = startPos;
        transform.localRotation = startRot;
        rBody.linearVelocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
        
        for (int i = 0; i < joints.Length; i++)
        {
            joints[i].transform.localPosition = jointStartPos[i];
            joints[i].transform.localRotation = jointStartRot[i];
            
            jointRbs[i].linearVelocity = Vector3.zero;
            jointRbs[i].angularVelocity = Vector3.zero;
            joints[i].targetAngularVelocity = Vector3.zero;
        }
        
        Physics.SyncTransforms();
        ResetTargetPosition();
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(TanhVector(transform.InverseTransformDirection(rBody.linearVelocity) / 10f));
        sensor.AddObservation(TanhVector(transform.InverseTransformDirection(rBody.angularVelocity) / 10f));
     
        for (int i = 0; i < joints.Length; i++)
        {
            Vector3 localAngVel = joints[i].transform.InverseTransformDirection(jointRbs[i].angularVelocity);
            float angVelX = localAngVel.x;
            sensor.AddObservation((float)System.Math.Tanh(angVelX / 10f));
        }

        Vector3 localTargetPos = transform.InverseTransformPoint(target.position);
        sensor.AddObservation(TanhVector(localTargetPos / 10f));
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        var continuousActions = actions.ContinuousActions; 
        for (int i = 0; i < joints.Length; i++)
        {
            float targetVel = continuousActions[i] * maxVelocity;
            joints[i].targetAngularVelocity = new Vector3(-targetVel, 0, 0);
        }

        Vector3 dirToTarget = (target.localPosition - transform.localPosition).normalized;
        float velocityTowardsTarget = Vector3.Dot(rBody.linearVelocity, dirToTarget);
        AddReward(velocityTowardsTarget * 0.01f);
        
        float disToTarget = Vector3.Distance(transform.localPosition, target.localPosition);           
        if (disToTarget < 0.75f)
        {
            AddReward(1.0f);
            ResetTargetPosition();
        }

        if (transform.localPosition.y < -1.0f)
        {
            AddReward(-1.0f);
            EndEpisode();
        }
    }
    
    private Vector3 TanhVector(Vector3 v)
    {
        return new Vector3((float)System.Math.Tanh(v.x), (float)System.Math.Tanh(v.y), (float)System.Math.Tanh(v.z));
    }
    
    private void ResetTargetPosition()
    {
        Vector3 randomPos = new Vector3(
            Random.Range(-targetRange, targetRange),
            target.localPosition.y,
            Random.Range(-targetRange, targetRange)
        );        
        target.localPosition = randomPos;
    }
}
