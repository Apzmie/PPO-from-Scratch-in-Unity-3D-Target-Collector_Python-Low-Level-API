using UnityEngine;
using TMPro;

public class ScoreManager : MonoBehaviour
{
    public static ScoreManager instance;
    public TextMeshProUGUI scoreText;

    private int playerScore = 0;
    private int aiScore = 0;

    private void Awake() 
    {
        instance = this; 
    }

    public void AddPlayerScore()
    {
        playerScore++;
        UpdateUI();
    }
    
    public void MinusPlayerScore()
    {
        playerScore--;
        UpdateUI();
    }

    public void AddAIScore()
    {
        aiScore++;
        UpdateUI();
    }
    
    public void MinusAIScore()
    {
        aiScore--;
        UpdateUI();
    }

    private void UpdateUI()
    {
        scoreText.text = $"Player: {playerScore}\nAI: {aiScore}";
    }
}
