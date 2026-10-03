/** Frontend runtime configuration.

`NEXT_PUBLIC_API_BASE_URL` must be set at build or run time. A default is
provided for local development but the production build will fail without it.
*/

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

if (!process.env.NEXT_PUBLIC_API_BASE_URL && process.env.NODE_ENV === "production") {
  // eslint-disable-next-line no-console
  console.warn(
    "NEXT_PUBLIC_API_BASE_URL is not set; the production build will use the default " +
      "which points to the local development server and will not work in production."
  );
}

export const DEFAULT_PAGE_SIZE = 50;
export const MAX_PAGE_SIZE = 500;