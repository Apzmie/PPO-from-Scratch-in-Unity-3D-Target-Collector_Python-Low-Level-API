using UnityEngine;
using UnityEngine.EventSystems;

public class Joystick : MonoBehaviour, IDragHandler, IPointerUpHandler, IPointerDownHandler
{
    public GameObject joystickVisual;
    public RectTransform handle;
    private RectTransform background;
    public Vector2 inputVector;

    private void Awake() 
    {
        background = joystickVisual.GetComponent<RectTransform>();
    }

    private void Start() 
    {
        joystickVisual.SetActive(false);
    }

    public void OnPointerDown(PointerEventData eventData) 
    {
        if (Input.touchCount > 0) 
        {
            joystickVisual.SetActive(true);
            OnDrag(eventData);
        }
    }

    public void OnDrag(PointerEventData eventData) 
    {
        if (!joystickVisual.activeSelf) return;

        Vector2 pos;
        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(background, eventData.position, eventData.pressEventCamera, out pos)) {
            pos.x = (pos.x / background.sizeDelta.x);
            pos.y = (pos.y / background.sizeDelta.y);

            inputVector = new Vector2(pos.x * 2, pos.y * 2);
            inputVector = (inputVector.magnitude > 1.0f) ? inputVector.normalized : inputVector;

            handle.anchoredPosition = new Vector2(inputVector.x * (background.sizeDelta.x / 2.5f), inputVector.y * (background.sizeDelta.y / 2.5f));
        }
    }

    public void OnPointerUp(PointerEventData eventData) 
    {
        inputVector = Vector2.zero;
        handle.anchoredPosition = Vector2.zero;
    }
}
