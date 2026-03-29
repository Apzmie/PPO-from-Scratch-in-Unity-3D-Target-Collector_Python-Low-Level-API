using UnityEngine;
using Unity.InferenceEngine;
using System.Threading.Tasks;

public class RunModel : MonoBehaviour
{
    public ModelAsset modelAsset;
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

    private Model runtimeModel;
    private Worker worker;
    private Tensor<float> inputTensor;
    private Tensor<float> cpuCopyTensor;
    private bool inferencing = false;
    
    private int decisionCounter = 0;
    private int decisionPeriod = 5;

    private void Start()
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

        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.CPU);
        
        ResetAgent();
        ResetTargetPosition();
    }

    private void FixedUpdate()
    {
        if (!inferencing)
        {
            if (decisionCounter == 0)
            {
                _ = InferenceAsync();
                decisionCounter = decisionPeriod;
            }
            decisionCounter--;
        }
        
        float disToTarget = Vector3.Distance(transform.localPosition, target.localPosition);           
        if (disToTarget < 0.75f)
        {
            ScoreManager.instance.AddAIScore();
            ResetTargetPosition();
        }
        
        if (transform.localPosition.y < -1.0f)
        {
            ResetAgent();
            ScoreManager.instance.MinusAIScore();
        }
    }

    private async Task InferenceAsync()
    {
        inferencing = true;
        
        float[] observation = new float[11];
        
        Vector3 linVel = TanhVector(transform.InverseTransformDirection(rBody.linearVelocity) / 10f);
        observation[0] = linVel.x;
        observation[1] = linVel.y;
        observation[2] = linVel.z;
        
        Vector3 angVel = TanhVector(transform.InverseTransformDirection(rBody.angularVelocity) / 10f);
        observation[3] = angVel.x;
        observation[4] = angVel.y;
        observation[5] = angVel.z;
        
        for (int i = 0; i < joints.Length; i++)
        {
            Vector3 localAngVel = joints[i].transform.InverseTransformDirection(jointRbs[i].angularVelocity);
            observation[6 + i] = (float)System.Math.Tanh(localAngVel.x / 10f);
        }
        
        Vector3 localTargetPos = TanhVector(transform.InverseTransformPoint(target.position) / 10f);
        observation[8] = localTargetPos.x;
        observation[9] = localTargetPos.y;
        observation[10] = localTargetPos.z;

        TensorShape shape = new TensorShape(1, 11);
        inputTensor?.Dispose();
        inputTensor = new Tensor<float>(shape, observation);
        worker.Schedule(inputTensor);
        using (var outputTensor = worker.PeekOutput() as Tensor<float>)
        {
            cpuCopyTensor = await outputTensor.ReadbackAndCloneAsync();
        }
        
        for (int i = 0; i < joints.Length; i++)
        {
            float actionValue = cpuCopyTensor[i];
            float targetVel = actionValue * maxVelocity;
            joints[i].targetAngularVelocity = new Vector3(-targetVel, 0, 0);
        }
        
        cpuCopyTensor?.Dispose();
        inferencing = false;
    }

    private Vector3 TanhVector(Vector3 v)
    {
        return new Vector3((float)System.Math.Tanh(v.x), (float)System.Math.Tanh(v.y), (float)System.Math.Tanh(v.z));
    }

    private void OnDestroy()
    {
        inputTensor?.Dispose();
        worker?.Dispose();
    }
    
    private void ResetTargetPosition()
    {
        Vector3 randomPos = new Vector3(
            UnityEngine.Random.Range(-targetRange, targetRange),
            target.localPosition.y,
            UnityEngine.Random.Range(-targetRange, targetRange)
        );        
        target.localPosition = randomPos;
    }
    
    public void ResetAgent()
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
        decisionCounter = 0;
    }
}
