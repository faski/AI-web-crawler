"""Crawler config for www.xe.com."""

from crawl4ai import CrawlerRunConfig

CONFIG = CrawlerRunConfig(
    magic=True,
    excluded_tags=["style", "script", "link", "meta", "noscript", "footer", "nav"],
    excluded_selector=(
        # Main site header (logo, nav, get-the-app button)
        "#siteHeader, "
        # Breadcrumb bar fixed below the header
        "div[class*='fixed inset-x-0'], "
        # Blog category navigation bar (Blog / Money Transfer / Personal Finance / etc.)
        "div[class*='border-[#ccc]'], "
        # Article hero image
        "img[alt='heading picture'], "
        # Author name + date/read-time row
        "div[class*='mb-8 mt-6 flex items-center'], "
        # Table of Contents box
        "div[class*='rounded-xl bg-gray-250'], "
        # Inline CTA / signup banner links embedded in article body
        "a[href*='account.xe.com'], "
        # CTA buttons (e.g. "Create a free business account", "Speak to an FX specialist")
        "[class*='polymorphic-btn'], "
        # Article tag chips at the bottom of the post
        "div[class*='flex flex-wrap gap-2'], "
        # Blue business-promo banner after the article body
        "section[itemtype*='BlogPosting'] + div, "
        # Related articles grid section
        "section[class*='py-12'], "
        # "Trusted by" partner-logos block on the converter page
        "div[data-testid='trusted-by'], "
        # Trustpilot reviews carousel on the converter page
        "div[data-testid='trustpilot-newcore'], "
        # Hidden sr-only "Click to change base currency to X"
        "span[id$='-aria-description'], "
        # "Last updated Apr DD, YYYY, HH:MM UTC" timestamp
        "div[data-testid='timestamp-template'], "
        # "Manage your currencies on the go with the Xe app"
        "div[class*='mb-24'][class*='from-blue-900'], "
        # Cookie / GDPR consent popup
        "div[class*='animate-slideFromLeft'], "
        # Inline Xe logo rendered as an SVG
        "svg[aria-label='Xe']"
    ),
    remove_forms=True,
)
