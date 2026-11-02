export const SIGNUP_CONFIRMATION_MESSAGE =
  "Account created. Please check your email and confirm your account before signing in.";

export function friendlyAuthError(error: { message: string; code?: string }): string {
  if (error.code === "email_not_confirmed" || /email not confirmed/i.test(error.message)) {
    return "Please confirm your email before signing in. Check your inbox for the confirmation link.";
  }
  if (error.code === "invalid_credentials" || /invalid login credentials/i.test(error.message)) {
    return "The email or password is incorrect. Please try again.";
  }
  if (/rate limit|too many requests/i.test(error.message)) {
    return "Too many attempts. Please wait a moment and try again.";
  }
  return "Authentication could not be completed. Please check your details and try again.";
}
