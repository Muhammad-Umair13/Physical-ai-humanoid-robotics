import { useEffect } from "react";

declare global {
  interface Window {
    Chatbot: any;
  }
}

const FrontendChatbot = () => {
  useEffect(() => {
    // Prevent double initialization
    if (window.Chatbot) return;

    const script = document.createElement("script");
    script.type = "module";
    script.defer = true;
    script.src = "https://cdn.n8nchatui.com/v1/embed.js";

    script.onload = () => {
      window.Chatbot.init({
        n8nChatUrl:
          "https://muhamadumair345.app.n8n.cloud/webhook/968bcb11-6ec7-460c-ba40-636bb53fcc58/chat",

        metadata: {},

        allowProgrammaticMessage: true,

        theme: {
          button: {
            backgroundColor: "#1E6BFF",
            right: 20,
            bottom: 20,
            size: 56,
            iconColor: "#FFFFFF",
            customIconSrc:
              "https://www.svgrepo.com/show/334455/bot.svg",
            customIconSize: 55,
            customIconBorderRadius: 50,
            borderRadius: "circle",
            autoWindowOpen: {
              autoOpen: false,
              openDelay: 8,
            },
          },

          tooltip: {
            showTooltip: true,
            tooltipMessage: "Hi 👋 Need help?",
            tooltipBackgroundColor: "#EEF2FF",
            tooltipTextColor: "#1F2937",
            tooltipFontSize: 14,
            hideTooltipOnMobile: true,
          },

          chatWindow: {
            borderRadiusStyle: "rounded",
            avatarBorderRadius: 50,
            messageBorderRadius: 14,
            showTitle: true,
            title: "Physical AI Assistant",
            avatarSize: 36,
            welcomeMessage:
              "Hi 👋 Ask me anything about Physical AI & Humanoid Robotics.",
            errorMessage:
              "Workflow is not connected yet. Please try again later.",
            backgroundColor: "#FFFFFF",
            height: 560,
            width: 380,
            fontSize: 15,

            starterPrompts: [
              "What is Physical AI?",
              "Explain Humanoid Robots",
              "What is ROS?",
            ],
            starterPromptFontSize: 14,

            renderHTML: true,
            clearChatOnReload: false,
            showScrollbar: true,

            botMessage: {
              backgroundColor: "#EEF2FF",
              textColor: "#1F2937",
              showAvatar: false,
              showCopyToClipboardIcon: false,
            },

            userMessage: {
              backgroundColor: "#1E6BFF",
              textColor: "#FFFFFF",
              showAvatar: true,
              avatarSrc:
                "https://www.svgrepo.com/show/532363/user-alt-1.svg",
            },

            textInput: {
              placeholder: "Type your message...",
              backgroundColor: "#FFFFFF",
              textColor: "#1F2937",
              sendButtonColor: "#1E6BFF",
              maxChars: 300,
              maxCharsWarningMessage:
                "Message is too long. Please shorten it.",
              autoFocus: true,
              borderRadius: 12,
              sendButtonBorderRadius: 50,
            },
          },
        },
      });
    };

    document.body.appendChild(script);

    return () => {
      script.remove();
    };
  }, []);

  return null;
};

export default FrontendChatbot;
