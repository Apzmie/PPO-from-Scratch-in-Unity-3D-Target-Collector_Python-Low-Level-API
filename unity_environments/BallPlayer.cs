using UnityEngine;
using UnityEngine.InputSystem;

public class BallPlayer : MonoBehaviour
{
    public Joystick joystick;
    public Transform target;
    
    private float speed = 5f;
    private Rigidbody rb;
    private float targetRange = 8f;
    private Vector3 initialPosition;

    private void Start()
    {
        rb = GetComponent<Rigidbody>();
        initialPosition = transform.localPosition;
    }

    private void FixedUpdate()
    {
        var keyboard = Keyboard.current;

        float moveX = 0;
        float moveZ = 0;

        if (keyboard.leftArrowKey.isPressed) moveX = -1f;
        else if (keyboard.rightArrowKey.isPressed) moveX = 1f;

        if (keyboard.upArrowKey.isPressed) moveZ = 1f;
        else if (keyboard.downArrowKey.isPressed) moveZ = -1f;
        
        if (joystick != null && joystick.inputVector != Vector2.zero)
        {
            moveX = joystick.inputVector.x;
            moveZ = joystick.inputVector.y;
        }

        Vector3 moveDirection = new Vector3(moveX, 0f, moveZ).normalized;
        if (moveDirection.magnitude > 1f)
        {
            moveDirection = moveDirection.normalized;
        }

        if (moveDirection != Vector3.zero)
        {
            Vector3 torque = new Vector3(moveDirection.z, 0, -moveDirection.x) * speed;
            rb.AddTorque(torque, ForceMode.Force);
        }
        
        float disToTarget = Vector3.Distance(transform.localPosition, target.localPosition);
        if (disToTarget < 1f)
        {
            ScoreManager.instance.AddPlayerScore();
            ResetTargetPosition();
        }
        
        if (transform.localPosition.y < -1.0f)
        {
            ResetPlayer();
            ScoreManager.instance.MinusPlayerScore();
        }
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
    
    private void ResetPlayer()
    {
        transform.localPosition = initialPosition;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
    }
}
