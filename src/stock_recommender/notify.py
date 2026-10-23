"""Email the recommendations through the SMTP server in the settings."""
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from html import escape
import logging
import smtplib
import ssl
from .config import Settings
from .schemas import EmailReport, Recommendation, RecommendationList

log = logging.getLogger("uvicorn.error")  # Uvicorn's general logger, which its console handler prints.
HEADERS = ["#", "Ticker", "Score", "Exp. return", "Sentiment", "News", "Close"]


def cells(item: Recommendation) -> list[str]:
    return [str(item.rank), item.ticker, f"{item.score:.4f}", f"{item.expected_return:+.2%}",
            f"{item.news_sentiment:+.2f}", str(item.news_count), f"{item.close:,.2f}"]


def compose(body: RecommendationList, sender: str, recipients: list[str]) -> EmailMessage:
    """Return a message listing the buys and sells, as plain text and as HTML tables."""
    intro = (f"Stocks to buy and sell at the next open, from bars through {body.as_of} and news published "
             f"before {body.news_cutoff:%Y-%m-%d %H:%M} UTC. Model {body.model_version}, "
             f"{body.horizon_days}-day horizon.")
    text, html = [intro], [f"<p>{escape(intro)}</p>"]
    for title, items in (("Buy", body.buy), ("Sell", body.sell)):
        rows = [cells(item) for item in items]
        widths = [max(map(len, column)) for column in zip(HEADERS, *rows)]
        text += ["", title.upper()] + ["  ".join(cell.rjust(width) for cell, width in zip(line, widths))
                                       for line in [HEADERS, *rows]]
        header = "".join(f"<th>{escape(cell)}</th>" for cell in HEADERS)
        lines = "".join("<tr>" + "".join(f'<td align="right">{escape(cell)}</td>' for cell in row) + "</tr>"
                        for row in rows)
        html.append(f'<h3>{title}</h3><table cellpadding="4"><tr>{header}</tr>{lines}</table>')
    message = EmailMessage()
    message["Subject"] = f"Stock recommendations as of {body.as_of}: {len(body.buy)} buy, {len(body.sell)} sell"
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid()
    message.set_content("\n".join(text))
    message.add_alternative(f"<html><body>{''.join(html)}</body></html>", subtype="html")
    return message


def send(body: RecommendationList, settings: Settings) -> EmailReport:
    """Email the recommendations to settings.email_to; a failure is reported, not raised."""
    recipients = [address.strip() for address in settings.email_to.split(",") if address.strip()]
    sender = settings.email_from or settings.smtp_user
    if not (settings.smtp_host and recipients and sender):
        return EmailReport(sent_to=[], error="Email is not configured: set RECOMMENDER_SMTP_HOST, "
                                             "RECOMMENDER_EMAIL_TO and RECOMMENDER_SMTP_USER or _EMAIL_FROM.")
    context = ssl.create_default_context()  # Verifies the server's certificate.
    try:
        message = compose(body, sender, recipients)
        if settings.smtp_port == 465:
            server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=30, context=context)
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)
        with server:
            server.ehlo()
            encrypted = settings.smtp_port == 465
            if not encrypted and server.has_extn("starttls"):
                server.starttls(context=context)
                server.ehlo()
                encrypted = True
            if settings.smtp_user:
                if not encrypted:
                    raise smtplib.SMTPException("The server offers no TLS; refusing to send the password unencrypted.")
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)
    except (OSError, smtplib.SMTPException) as error:
        log.error("Email failed: %s", error)
        return EmailReport(sent_to=[], error=f"{type(error).__name__}: {error}")
    except ValueError as error:  # Non-ASCII credentials or host, or a line break in an address.
        log.error("Email failed: invalid email settings (%s)", type(error).__name__)
        # The error's text can quote the password, so it is left out.
        return EmailReport(sent_to=[], error=f"{type(error).__name__}: invalid email settings; the SMTP host, user, "
                                             "password and addresses must be ASCII without line breaks.")
    return EmailReport(sent_to=recipients)
